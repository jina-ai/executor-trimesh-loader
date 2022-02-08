from pathlib import Path

import pytest
from jina import Document, DocumentArray

from executor import TrimeshLoader

data_dir = Path(__file__).parent


@pytest.fixture
def trimesh_loader():
    return TrimeshLoader()


@pytest.fixture
def document_uri():
    return Document(uri=str(data_dir / 'test.glb'))


@pytest.fixture
def document_base64():
    doc = Document(uri=str(data_dir / 'test.glb'), mime_type='application/octet-stream')
    doc.load_uri_to_buffer()
    doc.dump_buffer_to_datauri(base64=True)
    return doc


@pytest.fixture
def document_blob():
    doc = Document(uri=str(data_dir / 'test.glb'), mime_type='application/octet-stream')
    doc.load_uri_to_buffer()
    return doc


def test_doc_uri(trimesh_loader, document_uri):
    trimesh_loader.process(DocumentArray([document_uri]))
    assert document_uri.blob is not None
    assert document_uri.blob.shape == (1024, 3)


def test_doc_base64(trimesh_loader, document_base64):
    trimesh_loader.process(DocumentArray([document_base64]))
    assert document_base64.blob is not None
    assert document_base64.blob.shape == (1024, 3)


def test_doc_blob(trimesh_loader, document_blob):
    trimesh_loader.process(DocumentArray([document_blob]))
    assert document_blob.blob is not None
    assert document_blob.blob.shape == (1024, 3)
