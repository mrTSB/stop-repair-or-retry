import unittest

from textbook_corpus import exposition


class ExpositionOnly(unittest.TestCase):
    def test_assessments_and_worked_answers_are_excluded_but_formula_is_preserved(self):
        fact='A physical quantity can depend on the inverse square of distance between objects; this relationship describes the force between point charges.'
        raw=('<document><title>Point charges</title><content><para>'+fact+'</para>'
             '<equation><math><mfrac><mn>1</mn><msup><mi>r</mi><mn>2</mn></msup></mfrac></math></equation>'
             '<example><para>WORKED_ANSWER should never be included even if the paragraph is sufficiently long to pass a length cutoff.</para></example>'
             '<exercise><para>EXAM_ANSWER should never be included even if this paragraph also meets the ordinary content length threshold.</para></exercise>'
             '<para>Which option is correct? This long question must be excluded even when it is not enclosed by an explicit exercise element.</para>'
             '</content></document>').encode()
        title,paragraphs=exposition(raw)
        self.assertEqual(title,'Point charges')
        self.assertEqual(len(paragraphs),1)
        self.assertIn('(1)/((r)^(2))',paragraphs[0])
        self.assertNotIn('ANSWER',' '.join(paragraphs))


if __name__=='__main__':unittest.main()
