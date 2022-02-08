import pytest
from jina import Document, DocumentArray, Flow

from executor import TrimeshLoader


@pytest.fixture
def docs():
    docs = DocumentArray()
    doc = Document(
        uri='https://storage.googleapis.com/showcase-3d-models/ShapeNetV2/airplane_aeroplane_plane_0.glb'
    )
    return docs


def test_use_in_flow(docs):
    with Flow().add(uses=TrimeshLoader, name='trimesh_loader') as flow:
        data = flow.post(on='/index', inputs=docs, return_results=True)
        for doc in data[0].docs:
            assert doc.blob is not None
            assert doc.blob.shape == (1024, 3)