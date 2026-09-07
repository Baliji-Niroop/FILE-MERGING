import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from exam_merger.models.chunk import Chunk, ChunkGroup
from exam_merger.pipeline import normalizer, deduplicator, assembler
from exam_merger.extractors import pdf_extractor
from exam_merger.diagrams import vision_describer


class TestNormalizer(unittest.TestCase):

    def test_normalize_quotes_and_whitespace(self):
        """Smart quotes are converted to ASCII; triple blank lines collapse to one."""
        chunks = [
            # \u201c\u201d are smart double quotes — cleaned to ASCII "
            Chunk(text='  \u201cHello\u201d World!  ', source_file='doc1.docx',
                  location='page 1', chunk_type='text'),
            Chunk(text='page 12', source_file='doc1.docx',
                  location='page 12', chunk_type='text'),
            Chunk(text='15', source_file='doc1.docx',
                  location='page 15', chunk_type='text'),
            Chunk(text='Some text.\n\n\nMore text.', source_file='doc1.docx',
                  location='page 1', chunk_type='text'),
        ]

        normalized = normalizer.normalize(chunks)

        # "page 12" and "15" are boilerplate patterns; "\u201cHello\u201d World!" is too
        # short (14 chars < MIN_CHUNK_CHARS=40) so 3 are dropped, leaving only
        # "Some text.\n\nMore text." which is 22+ chars.
        # BUT "\u201cHello\u201d World!" has 14 chars — below MIN_CHUNK_CHARS.
        # So only the long one survives.
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].text, 'Some text.\n\nMore text.')

    def test_normalize_keeps_tables_regardless_of_length(self):
        """Tables shorter than MIN_CHUNK_CHARS must still be kept."""
        chunks = [
            Chunk(text='| A | B |\n| --- | --- |\n| 1 | 2 |',
                  source_file='doc.docx', location='Table', chunk_type='table'),
        ]
        normalized = normalizer.normalize(chunks)
        self.assertEqual(len(normalized), 1)

    def test_normalize_removes_stub_fragments(self):
        """Single-line fragments below threshold are removed for text chunks."""
        chunks = [
            Chunk(text='See Figure 3', source_file='doc.docx',
                  location='page 1', chunk_type='text'),
            Chunk(text='This is a long enough chunk of text that should survive the normalizer filter easily.',
                  source_file='doc.docx', location='page 2', chunk_type='text'),
        ]
        normalized = normalizer.normalize(chunks)
        self.assertEqual(len(normalized), 1)
        self.assertIn('long enough', normalized[0].text)

    def test_normalize_unicode_nfkc(self):
        """Ligatures and fullwidth characters are NFKC-normalised."""
        chunks = [
            # \ufb01 = fi ligature; NFKC → 'fi'
            Chunk(text='\ufb01nancial statements are important for understanding a company\'s \ufb01nancial health.',
                  source_file='doc.docx', location='page 1', chunk_type='text'),
        ]
        normalized = normalizer.normalize(chunks)
        self.assertEqual(len(normalized), 1)
        self.assertIn('financial', normalized[0].text)
        self.assertNotIn('\ufb01', normalized[0].text)


class TestDeduplicator(unittest.TestCase):

    @patch('exam_merger.pipeline.deduplicator.get_embedding_model')
    def test_deduplicate_near_identical_discard(self, mock_get_model):
        """Near-duplicate chunks (sim >= 0.95) — only the longer one is kept."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # A and B are near-identical (normalised dot ~ 0.97 >= 0.95)
        # C is unrelated
        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.97, 0.24, 0.0], dtype=np.float32)
        emb_c = np.array([0.3, 0.0, 0.95], dtype=np.float32)

        mock_model.encode.return_value = [emb_a, emb_b, emb_c]

        chunks = [
            Chunk(text="Newton's second law: F = ma",
                  source_file='file1.docx', location='page 1', chunk_type='text'),
            Chunk(text="Newton's Second Law of Motion: Net force equals mass times acceleration (F = ma).",
                  source_file='file2.pptx', location='slide 3', chunk_type='text'),
            Chunk(text='Gravity is a force that pulls objects toward each other.',
                  source_file='file1.docx', location='page 2', chunk_type='text'),
        ]

        groups = deduplicator.deduplicate(chunks, threshold=0.82)

        self.assertEqual(len(groups), 2)

        group_texts = [g.representative_text for g in groups]
        # B is longer than A, so only B is kept as representative
        self.assertIn("Newton's Second Law of Motion: Net force equals mass times acceleration (F = ma).",
                      group_texts)
        self.assertNotIn("Newton's second law: F = ma", group_texts)
        self.assertIn('Gravity is a force that pulls objects toward each other.',
                      group_texts)

    @patch('exam_merger.pipeline.deduplicator.get_embedding_model')
    def test_deduplicate_concatenate_related_distinct(self, mock_get_model):
        """Related-but-distinct chunks (threshold <= sim < 0.95) are merged with Additional Detail."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb_b = np.array([0.88, 0.47, 0.0], dtype=np.float32)

        mock_model.encode.return_value = [emb_a, emb_b]

        chunks = [
            Chunk(text='Summary: force equals mass times acceleration.',
                  source_file='file1.pptx', location='slide 2', chunk_type='text'),
            Chunk(text='Detailed worked example: a box of 10kg is pushed with 50N of force, resulting in 5 m/s^2 acceleration.',
                  source_file='file2.pdf', location='page 5', chunk_type='text'),
        ]

        groups = deduplicator.deduplicate(chunks, threshold=0.82)

        self.assertEqual(len(groups), 1)
        rep_text = groups[0].representative_text
        self.assertIn('Summary: force equals mass times acceleration.', rep_text)
        self.assertIn('Detailed worked example', rep_text)
        self.assertIn('**Additional Detail:**', rep_text)

    @patch('exam_merger.pipeline.deduplicator.get_embedding_model')
    def test_deduplicate_all_unique(self, mock_get_model):
        """Orthogonal chunks (sim ~ 0) form separate groups."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        mock_model.encode.return_value = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
        ]

        chunks = [
            Chunk(text='Topic A: quantum entanglement.', source_file='a.pdf',
                  location='page 1', chunk_type='text'),
            Chunk(text='Topic B: machine learning optimisation.', source_file='b.pdf',
                  location='page 1', chunk_type='text'),
            Chunk(text='Topic C: organic chemistry reactions.', source_file='c.pdf',
                  location='page 1', chunk_type='text'),
        ]

        groups = deduplicator.deduplicate(chunks, threshold=0.82)
        self.assertEqual(len(groups), 3)


class TestAssembler(unittest.TestCase):

    def test_assemble_with_toc_and_citations(self):
        chunks_1 = [
            Chunk(text="Newton's second law: F=ma",
                  source_file='file1.docx',
                  location='Section: Physics > Dynamics',
                  chunk_type='text')
        ]
        chunks_2 = [
            Chunk(text='Table content',
                  source_file='file2.pdf',
                  location='page 1',
                  chunk_type='table')
        ]

        groups = [
            ChunkGroup(representative_text="Newton's second law: F=ma",
                       members=chunks_1, topic_tag='Physics'),
            ChunkGroup(representative_text='Table content',
                       members=chunks_2, topic_tag='file2'),
        ]

        output = assembler.assemble(
            groups, ['/path/to/physics/file1.docx', '/path/to/physics/file2.pdf']
        )

        self.assertIn('# physics', output.lower())
        self.assertIn('*Generated from: file1.docx, file2.pdf*', output)
        self.assertIn('*Generated on:', output)  # timestamp present
        self.assertIn('## Physics', output)
        self.assertIn("Newton's second law: F=ma", output)
        self.assertIn('*Sources: file1.docx (Section: Physics > Dynamics)*', output)

    def test_assemble_case_insensitive_topic_dedup(self):
        """Groups with the same topic in different cases collapse into one section."""
        chunks = [
            Chunk(text='First chunk about physics.',
                  source_file='f1.docx', location='Section: Physics', chunk_type='text'),
            Chunk(text='Second chunk about physics.',
                  source_file='f2.docx', location='Section: physics', chunk_type='text'),
        ]

        groups = [
            ChunkGroup(representative_text='First chunk about physics.',
                       members=[chunks[0]], topic_tag='Physics'),
            ChunkGroup(representative_text='Second chunk about physics.',
                       members=[chunks[1]], topic_tag='physics'),
        ]

        output = assembler.assemble(groups, ['/tmp/f1.docx', '/tmp/f2.docx'])

        # There should be exactly one ## Physics section
        physics_count = output.count('\n## Physics') + output.count('\n## physics')
        self.assertEqual(physics_count, 1)

    def test_assemble_breaks_topic_ties_deterministically(self):
        chunks = [
            Chunk(text='History', source_file='file1.docx',
                  location='Heading 1', chunk_type='text'),
            Chunk(text='Physics', source_file='file2.docx',
                  location='Heading 1', chunk_type='text'),
        ]
        groups = [
            ChunkGroup(representative_text='History',
                       members=[chunks[0], chunks[1]])
        ]

        output = assembler.assemble(groups, ['/tmp/course/file1.docx', '/tmp/course/file2.docx'])

        self.assertIn('## History', output)
        self.assertNotIn('## Physics', output)


class TestPdfExtractor(unittest.TestCase):

    def test_pdf_extract_splits_long_single_paragraph(self):
        long_text = 'A' * 20050

        class FakePage:
            def __init__(self, text=None):
                self._text = text

            def extract_text(self):
                return self._text

            def extract_tables(self):
                return []

        class FakePdf:
            pages = [FakePage()]

        class FakePdfplumberContext:
            def __enter__(self):
                self.pages = [FakePage(text='A' * 20050)]
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def open(self, *a, **kw):
                return self

        with patch('exam_merger.extractors.pdf_extractor.pdfplumber', None), \
             patch('exam_merger.extractors.pdf_extractor.pypdf.PdfReader',
                   return_value=FakePdf()):
            with patch.object(FakePage, 'extract_text', return_value=long_text):
                chunks = pdf_extractor.extract('/tmp/fake.pdf')

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 10000 for chunk in chunks))


class TestVisionDescriber(unittest.TestCase):

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('exam_merger.diagrams.vision_describer.anthropic.Anthropic')
    def test_describe_image_uses_supplied_media_type(self, mock_anthropic):
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text='diagram summary')]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        result = vision_describer.describe_image(b'fake-bytes', media_type='image/jpeg')

        self.assertEqual(result, 'diagram summary')
        called_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(
            called_kwargs['messages'][0]['content'][0]['source']['media_type'],
            'image/jpeg'
        )


if __name__ == '__main__':
    unittest.main()
