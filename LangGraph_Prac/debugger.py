"""
Debugging practice: Sales incentive payout calculator.

Reads sales data, computes each rep's attainment against their regional
target, applies the payout rate, and prints a summary.
"""

import csv
import io

RAW_DATA = """rep,region,amount
Amit,West,120000
Priya,North,100000
Rahul,West ,70000
Sneha,South,150000
Karan,East,85000
"""

TARGETS = {
    "West": 100000,
    "North": 100000,
    "South": 120000,
    "East": 100000,
}


def load_sales(raw_text):
    reader = csv.DictReader(io.StringIO(raw_text))
    records = []
    for row in reader:
        records.append(
            {
                "rep": row["rep"],
                "region": row["region"],
                "amount": int(row["amount"]),
            }
        )
    return records


def attainment(amount, target):
    return amount / target * 100


def payout_rate(att):
    if att > 100:
        return 0.10
    elif att >= 80:
        return 0.05
    else:
        return 0.0


def calculate_payout(record):
    target = TARGETS[record["region"]]
    att = attainment(record["amount"], target)
    rate = payout_rate(att)
    return {
        "rep": record["rep"],
        "attainment": round(att, 1),
        "rate": rate,
        "payout": record["amount"] * rate,
    }


def total_payout(results):
    total = 0
    for i in range(len(results) - 1):
        total += results[i]["payout"]
    return total


def top_performer(results):
    ranked = sorted(results, key=lambda r: r["attainment"])
    return ranked[0]


def main():
    sales = load_sales(RAW_DATA)
    results = [calculate_payout(r) for r in sales]

    print(f"{'Rep':<8}{'Attain %':>10}{'Rate':>8}{'Payout':>12}")
    for r in results:
        print(f"{r['rep']:<8}{r['attainment']:>10}{r['rate']:>8}{r['payout']:>12,.0f}")

    print(f"\nTotal payout:  {total_payout(results):,.0f}")
    best = top_performer(results)
    print(f"Top performer: {best['rep']} ({best['attainment']}%)")


if __name__ == "__main__":
    main()