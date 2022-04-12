import os
import tempfile
import urllib
from typing import Dict, Optional

import numpy as np
import trimesh
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger

# hack to ignore loading image materials for gltf
trimesh.exchange.gltf._parse_materials = lambda *args, **kwargs: None


class TrimeshLoader(Executor):
    """An Executor for loading triangular meshes and extract point cloud"""

    def __init__(
        self,
        samples: int = 1024,
        as_chunks: bool = False,
        drop_content: bool = True,
        *args,
        **kwargs,
    ):
        """
        :param samples: default number of points to sample from the mesh
        :param as_chunks: when multiple geometry stored in one mesh file,
            then store each geometry into different :attr:`.chunks`
        :param drop_content: if True, the content of document (`uri` or `blob`) will be dropped in the end.
        :param args: the *args for Executor
        :param kwargs: the **kwargs for Executor
        """
        super().__init__(*args, **kwargs)
        self.samples = samples
        self.as_chunk = as_chunks
        self.drop_content = drop_content
        self.logger = JinaLogger(
            getattr(self.metas, 'name', self.__class__.__name__)
        ).logger

    @requests
    def process(self, docs: DocumentArray, parameters: Optional[Dict] = {}, **kwargs):
        """Convert a 3d mesh-like :attr:`.uri` into :attr:`.tensor`"""
        if docs is None:
            return

        as_chunks = bool(parameters.get('as_chunks', self.as_chunk))
        samples = parameters.get('samples', self.samples)

        for doc in docs:
            if not doc.uri and doc.content is None:
                self.logger.error(
                    f'No uri or content passed for the Document: {doc.id}'
                )
                continue

            try:
                tmp_file = None
                if doc.uri:
                    schema = urllib.parse.urlparse(doc.uri).scheme
                    if schema in ['data', 'http', 'https']:
                        if schema in ['http', 'https']:
                            self._load(
                                doc,
                                doc.uri,
                                samples,
                                is_remote=True,
                                as_chunks=as_chunks,
                            )
                        elif schema == 'data':
                            # the default format is `glb`
                            file_format = doc.tags.get('file_format', 'glb')

                            tmp_file = tempfile.NamedTemporaryFile(
                                suffix=f'.{file_format}', delete=False
                            )
                            doc.load_uri_to_blob()
                            doc.save_blob_to_file(tmp_file.name)

                            if file_format == 'zip':
                                self._load_zip(
                                    doc, tmp_file.name, samples, as_chunks=as_chunks
                                )
                            else:
                                self._load(
                                    doc,
                                    tmp_file.name,
                                    samples,
                                    is_remote=False,
                                    as_chunks=as_chunks,
                                )

                            if self.drop_content:
                                doc.pop('uri')
                    elif doc.uri.endswith('.zip'):
                        self._load_zip(doc, doc.uri, samples, as_chunks=as_chunks)
                    else:
                        self._load(
                            doc, doc.uri, samples, is_remote=False, as_chunks=as_chunks
                        )

                elif doc.blob:
                    # the default format is `glb`
                    file_format = doc.tags.get('file_format', 'glb')
                    tmp_file = tempfile.NamedTemporaryFile(
                        suffix=f'.{file_format}', delete=False
                    )
                    doc.save_blob_to_file(tmp_file.name)
                    self._load(
                        doc,
                        tmp_file.name,
                        samples,
                        is_remote=False,
                        as_chunks=as_chunks,
                    )
                    if self.drop_content:
                        doc.pop('blob')
                else:
                    continue

                if tmp_file:
                    tmp_file.close()
                    os.unlink(tmp_file.name)
            except Exception as ex:
                self.logger.warning(
                    f'Will ignore the doc (uri={doc.uri}) with the exception: {ex!r}.'
                )
                continue

        return DocumentArray(
            d
            for d in docs
            if (len(d.chunks) > 0 if as_chunks else (d.tensor is not None))
        )

    def _load_zip(self, doc, uri, samples: int, as_chunks: bool = False):
        import shutil
        import zipfile
        from itertools import chain
        from pathlib import Path

        zf = zipfile.ZipFile(uri)
        target = Path(uri + '.extracted')

        try:
            zf.extractall(path=target)

            for file in list(chain(target.glob('**/*.gltf'), target.glob('**/*.glb'))):
                self._load(
                    doc,
                    str(file),
                    samples,
                    is_remote=False,
                    as_chunks=as_chunks,
                )
                break

        except Exception as ex:
            raise ex
        finally:
            zf.close()
            shutil.rmtree(target)

    def _load(
        self, doc, uri, samples: int, is_remote: bool = False, as_chunks: bool = False
    ):

        if as_chunks:
            # try to coerce everything into a scene
            if is_remote:
                scene = trimesh.load_remote(uri, force='scene')
            else:
                scene = trimesh.load(uri, force='scene')

            for geo in scene.geometry.values():
                geo: trimesh.Trimesh
                geo_samples = geo.sample(samples)
                if np.isnan(geo_samples).any():
                    raise ValueError('NaN values contained in the model')
                doc.chunks.append(Document(tensor=geo_samples))
        else:
            # combine a scene into a single mesh
            if is_remote:
                mesh = trimesh.load_remote(uri, force='mesh')
            else:
                mesh = trimesh.load(uri, force='mesh')

            mesh_samples = mesh.sample(samples)
            if np.isnan(mesh_samples).any():
                raise ValueError('NaN values contained in the model')
            doc.tensor = mesh_samples
