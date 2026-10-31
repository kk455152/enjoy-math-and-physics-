"""Isolated synthetic corruption checks for native exported field artifacts."""
import tempfile
from pathlib import Path
import unittest
from full_transient_stepper import BACKEND_DIR
from comsol_backend import check_field_snapshot, read_force_history


class ExportChecks(unittest.TestCase):
    def test_field_export_rejects_empty_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic_fields.csv"
            path.write_text("% synthetic isolated test data\n",encoding="utf-8")
            with self.assertRaises(ValueError):check_field_snapshot(path)
            path.write_text(",".join(["0"]*20+["NaN"])+"\n",encoding="utf-8")
            with self.assertRaises(ValueError):check_field_snapshot(path)
            path.write_text(",".join(["0"]*21)+"\n",encoding="utf-8")
            self.assertEqual(check_field_snapshot(path),1)

    def test_force_table_rejects_corrupted_or_nonfinite_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"synthetic_forces.txt"
            path.write_text("% t_s Fx_total_N\n0 1 2\n",encoding="utf-8")
            with self.assertRaises(ValueError):read_force_history(path)
            path.write_text("% t_s Fx_total_N\n0 inf\n",encoding="utf-8")
            with self.assertRaises(ValueError):read_force_history(path)


if __name__=="__main__":unittest.main()
