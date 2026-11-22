import unittest
import broad_textbook_corpus as corpus


class BroadCorpusTests(unittest.TestCase):
    def test_answer_keys_and_third_party_papers_are_excluded(self):
        for url,title in (('/resources/08_lec_8_06_ans.pdf','PDF'),('/resources/x','Answers (PDF)'),
                          ('/resources/prob_sol_ses_key.pdf','PDF'),('/resources/exam1.pdf','PDF'),
                          ('/resources/MIT8_07F12_ln13.pdf','PDF'),('/resources/pset1.pdf','PDF')):
            self.assertFalse(corpus.allowed_resource(url,title))
        self.assertTrue(corpus.allowed_resource('/resources/lec25.pdf','Enols and enolates (PDF)'))
        self.assertTrue(corpus.allowed_resource('/resources/lec_ch7.pdf','Radioactive decay (PDF)'))

    def test_fetch_scope_is_official_course_assets_only(self):
        self.assertEqual(corpus.checked_url('https://ocw.mit.edu/courses/a/resource.pdf'),'https://ocw.mit.edu/courses/a/resource.pdf')
        for url in ('http://ocw.mit.edu/courses/a.pdf','https://example.com/a.pdf','https://ocw.mit.edu/other/a.pdf'):
            with self.assertRaises(ValueError):corpus.checked_url(url)


if __name__=='__main__':unittest.main()
