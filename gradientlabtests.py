import unittest
from ipynb.fs.full.gradientDescentLab import (error_at_point, shows, b_gradient, m_gradient, gradient_step, gradient_steps)

class TestDistance(unittest.TestCase):
    def test_regression_formula(self):
        m = 2.2
        b = 10
        budget = 20
        self.assertEqual(error_at_point(budget, m, b, shows), -4.0)

    def test_b_gradient(self):
        m = 2.2
        b = 10
        self.assertEqual(b_gradient(m, b, shows), 12.333333333333332)
        b_gradient(m, b, shows)

    def test_m_gradient(self):
        m = 2.2
        b = 10
        self.assertEqual(m_gradient(m, b, shows), 416.66666666666663)

    def test_gradient_step(self):
        learning_rate = .0001
        m = 2.2
        b = 10
        self.assertEqual(gradient_step(m, b, shows, learning_rate), {'b': 9.998766666666667, 'm': 2.1583333333333337})

    def test_gradient_steps(self):
         learning_rate = .0001
         initial_m = 0
         initial_b = 0
         steps = gradient_steps(initial_m, initial_b, shows, learning_rate, 1000)
         last_three_steps = steps[-3:]
         steps = [{'b': 0.4356103167800832, 'm': 1.9237500396834004, 'rss': 231.20511575689545},
         {'b': 0.43597800565581063,
          'm': 1.9237377794997421,
          'rss': 231.19699512173926},
         {'b': 0.43634568636974563, 'm': 1.92372551958823, 'rss': 231.1888748470956}]
         self.assertEqual(last_three_steps, steps)
if __name__ == '__main__':
    unittest.main()
