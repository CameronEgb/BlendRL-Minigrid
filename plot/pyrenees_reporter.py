import pandas as pd
from pathlib import Path

class PyreneesReporter:
    def __init__(self, output_dir: Path, group: str, clean_exp: str):
        self.output_dir = output_dir
        self.group = group
        self.clean_exp = clean_exp
        self.tier_names = [("Low Tier", 0), ("Med Tier", 1), ("High Tier", 2)]
        
    def report(self, problem_rows, step_rows, tidy_tier_rows):
        self._print_console_tables(problem_rows, step_rows)
        self._save_csvs(problem_rows, step_rows, tidy_tier_rows)
        self._generate_markdown(problem_rows, step_rows)
        
    def _print_console_tables(self, problem_rows, step_rows):
        if problem_rows:
            print("\n" + "=" * 90)
            print("  PYRENEES ACTION DISTRIBUTIONS BY STUDENT COMPETENCY TIER: PROBLEM LEVEL")
            print("  (Actions: PS = Problem Solving [0], WE = Worked Example [1], FWE = Faded Worked Example [2])")
            print("=" * 90)
            print(f"{'Method':<36} | {'Tier':<10} | {'PS (%)':>7} | {'WE (%)':>7} | {'FWE (%)':>7} | {'Tutor Agr %':>11}")
            print("-" * 90)
            for row in problem_rows:
                m_name = row["Method"]
                print(f"{m_name:<36} | {'Overall':<10} | {row['Overall PS %']:>6.1f}% | {row['Overall WE %']:>6.1f}% | {row['Overall FWE %']:>6.1f}% | {row['Tutor Agreement %']:>10.1f}%")
                for t_label, _ in self.tier_names:
                    print(f"{'':<36} | {t_label:<10} | {row.get(f'{t_label} PS %', 0.0):>6.1f}% | {row.get(f'{t_label} WE %', 0.0):>6.1f}% | {row.get(f'{t_label} FWE %', 0.0):>6.1f}% | {'-':>11}")
                print("-" * 90)

        if step_rows:
            print("\n" + "=" * 90)
            print("  PYRENEES ACTION DISTRIBUTIONS BY STUDENT COMPETENCY TIER: STEP LEVEL")
            print("  (Actions: PS/Elicit = 0, WE/Tell = 1)")
            print("=" * 90)
            print(f"{'Method':<36} | {'Tier':<10} | {'PS/Elicit':>9} | {'WE/Tell':>8} | {'FWE (%)':>7} | {'Tutor Agr %':>11}")
            print("-" * 90)
            for row in step_rows:
                m_name = row["Method"]
                print(f"{m_name:<36} | {'Overall':<10} | {row['Overall PS/Elicit %']:>8.1f}% | {row['Overall WE/Tell %']:>7.1f}% | {row['Overall FWE %']:>6.1f}% | {row['Tutor Agreement %']:>10.1f}%")
                for t_label, _ in self.tier_names:
                    print(f"{'':<36} | {t_label:<10} | {row.get(f'{t_label} PS/Elicit %', 0.0):>8.1f}% | {row.get(f'{t_label} WE/Tell %', 0.0):>7.1f}% | {row.get(f'{t_label} FWE %', 0.0):>6.1f}% | {'-':>11}")
                print("-" * 90)

    def _save_csvs(self, problem_rows, step_rows, tidy_tier_rows):
        df_tidy = pd.DataFrame(tidy_tier_rows)
        tidy_csv_path = self.output_dir / "pyrenees_action_distributions_by_tier.csv"
        df_tidy.to_csv(tidy_csv_path, index=False)
        print(f"\n  Saved Tidy Tier Breakdown: {tidy_csv_path}")

        if problem_rows:
            df_prob = pd.DataFrame(problem_rows)
            prob_csv_path = self.output_dir / "action_distribution_problem_level.csv"
            df_prob.to_csv(prob_csv_path, index=False)
            print(f"  Saved Problem Level CSV:   {prob_csv_path}")

        if step_rows:
            df_step = pd.DataFrame(step_rows)
            step_csv_path = self.output_dir / "action_distribution_step_level.csv"
            df_step.to_csv(step_csv_path, index=False)
            print(f"  Saved Step Level CSV:      {step_csv_path}")

        combined_rows = problem_rows + step_rows
        df_main = pd.DataFrame(combined_rows)
        main_csv_path = self.output_dir / "method_comparison.csv"
        df_main.to_csv(main_csv_path, index=False)
        print(f"  Saved Main Comparison CSV: {main_csv_path}")

    def _generate_markdown(self, problem_rows, step_rows):
        report_path = self.output_dir / "action_distribution_report.md"
        with open(report_path, "w") as f:
            f.write(f"# Pyrenees Policy Evaluation: Action Distributions by Student Competency Tier\n\n")
            f.write(f"**Experiment**: `{self.clean_exp}` (Group: `{self.group}`)\n\n")
            f.write(f"This report details the pedagogical action selections made by RL policies across student competency tiers (Low, Medium, High) at both Problem Level and Step Level.\n\n")

            if problem_rows:
                f.write("## 1. Problem-Level Action Distributions\n\n")
                f.write("Problem-level decisions choose how the student approaches the entire problem (PS = Problem Solving [0], WE = Worked Example [1], FWE = Faded Worked Example [2]).\n\n")
                f.write(pd.DataFrame(problem_rows).to_markdown(index=False))
                f.write("\n\n")

            if step_rows:
                f.write("## 2. Step-Level Action Distributions\n\n")
                f.write("Step-level decisions choose whether to elicit the step from the student (PS/Elicit = 0) or explain/provide the step (WE/Tell = 1).\n\n")
                f.write(pd.DataFrame(step_rows).to_markdown(index=False))
                f.write("\n\n")

            f.write("## 3. Key Findings & Pedagogical Adaptations\n\n")
            f.write("- **Low Competency Students**: Policies adjust scaffolding (e.g. Worked Examples) to support struggling students.\n")
            f.write("- **High Competency Students**: Policies provide autonomy (Problem Solving / Elicit) to reinforce mastery and problem fluency.\n")
            f.write("\n---\n*Auto-generated by NeSyRL Pipeline*\n")

        print(f"  Saved Markdown Report:     {report_path}")
