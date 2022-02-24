import os
import tempfile
import urllib
from typing import Dict, Optional

import trimesh
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


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

            tmp_file = None
            if doc.uri:
                schema = urllib.parse.urlparse(doc.uri).scheme
                if schema in ['data', 'http', 'https']:
                    if schema in ['http', 'https']:
                        self._load(
                            doc, doc.uri, samples, is_remote=True, as_chunks=as_chunks
                        )
                    elif schema == 'data':
                        # the default format is `glb`
                        file_format = doc.tags.get('file_format', 'glb')

                        tmp_file = tempfile.NamedTemporaryFile(
                            suffix=f'.{file_format}', delete=False
                        )
                        doc.load_uri_to_blob()
                        doc.save_blob_to_file(tmp_file.name)

                        self._load(
                            doc,
                            tmp_file.name,
                            samples,
                            is_remote=False,
                            as_chunks=as_chunks,
                        )

                        if self.drop_content:
                            doc.pop('uri')
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
                    doc, tmp_file.name, samples, is_remote=False, as_chunks=as_chunks
                )
                if self.drop_content:
                    doc.pop('blob')
            else:
                continue

            if tmp_file:
                os.unlink(tmp_file.name)

        return DocumentArray(
            d
            for d in docs
            if (len(d.chunks) > 0 if as_chunks else (d.tensor is not None))
        )

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
                doc.chunks.append(Document(tensor=geo.sample(samples)))
        else:
            # combine a scene into a single mesh
            if is_remote:
                mesh = trimesh.load_remote(uri, force='mesh')
            else:
                mesh = trimesh.load(uri, force='mesh')
            doc.tensor = mesh.sample(samples)
