"""
  pretty.py - rich console printing
"""

from rich.console import Console
from rich.table import Table

console = Console()

def print_results(title, results):
    """ results: {method_name: {metric_name: float}} -- columns are the union of metric names """
    cols  = list(dict.fromkeys(k for r in results.values() for k in r))
    best  = max(r['acc'] for r in results.values())
    table = Table(title=title, title_style='bold')
    table.add_column('method')
    for c in cols:
        table.add_column(c, justify='right', style='dim' if c == 'time' else None)
    
    for name, r in results.items():
        row = [f'{r[c]:0.5f}' if c in r else '-' for c in cols]
        table.add_row(name, *row, style='bold green' if r['acc'] == best else None)
    
    console.print(table)
