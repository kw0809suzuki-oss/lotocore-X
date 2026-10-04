from __future__ import annotations
import unittest
from compress60_to40_threeway import assembler40

class Compress60To40Test(unittest.TestCase):
    def test_assembler_returns_40_and_20_from_each_half(self):
        a=[[1,2,3,4,5,6,7] for _ in range(30)]
        b=[[31,32,33,34,35,36,37] for _ in range(30)]
        out=assembler40(a,b)
        self.assertEqual(len(out),40)
        self.assertEqual(sum(t==a[0] for t in out),20)
        self.assertEqual(sum(t==b[0] for t in out),20)

if __name__=="__main__":
    unittest.main()
