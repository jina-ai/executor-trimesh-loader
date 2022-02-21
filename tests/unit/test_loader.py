from pathlib import Path

import pytest
from jina import Document, DocumentArray

from executor import TrimeshLoader

data_dir = Path(__file__).parent.parent


@pytest.fixture
def trimesh_loader():
    return TrimeshLoader()


@pytest.fixture
def document_uri():
    return Document(uri=str(data_dir / 'test.glb'))


@pytest.fixture
def document_base64():
    doc = Document(uri=open(data_dir / 'test.base64').read())
    return doc


@pytest.fixture
def document_blob():
    doc = Document(uri=str(data_dir / 'test.glb'), mime_type='application/octet-stream')
    doc.load_uri_to_blob()
    return doc


def test_doc_uri(trimesh_loader, document_uri):
    trimesh_loader.process(DocumentArray([document_uri]))
    assert document_uri.tensor is not None
    assert document_uri.tensor.shape == (1024, 3)


def test_doc_base64(trimesh_loader, document_base64):
    trimesh_loader.process(DocumentArray([document_base64]))
    assert document_base64.tensor is not None
    assert document_base64.tensor.shape == (1024, 3)


def test_doc_blob(trimesh_loader, document_blob):
    trimesh_loader.process(DocumentArray([document_blob]))
    assert document_blob.tensor is not None
    assert document_blob.tensor.shape == (1024, 3)
