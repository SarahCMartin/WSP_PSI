#PSI_appendix

import os
import re
from collections import defaultdict
from datetime import datetime

# Set your folder path here
folder_path = 'your/figures/folder'

# Example input data for the 5%, 50%, 95% values
# Populate these dictionaries with your actual values
input_percentiles = {
    'Input1': (1.0, 5.0, 9.0),
    'Input2': (2.0, 6.0, 10.0),
}

output_percentiles = {
    'Output1': (3.0, 7.0, 11.0),
    'Output2': (4.0, 8.0, 12.0),
}

# Gather files
files = os.listdir(folder_path)

# Regex patterns
cdf_pdf_pattern = re.compile(r'(.+?)_(.+?)_CDF_and_PDF_(\d+)\.png')
correlation_pattern = re.compile(r'(.+?)_to_(.+?)_Correlation_(\d+)\.png')

# Categorize files
input_figures = {}
output_figures = {}
correlation_figures = defaultdict(list)

for f in files:
    cdf_pdf_match = cdf_pdf_pattern.match(f)
    corr_match = correlation_pattern.match(f)

    if cdf_pdf_match:
        var, dist, timestamp = cdf_pdf_match.groups()
        timestamp = int(timestamp)
        if var in input_percentiles:
            if var not in input_figures or timestamp > input_figures[var][1]:
                input_figures[var] = (f, timestamp)
        elif var in output_percentiles:
            if var not in output_figures or timestamp > output_figures[var][1]:
                output_figures[var] = (f, timestamp)

    elif corr_match:
        var1, var2, timestamp = corr_match.groups()
        timestamp = int(timestamp)
        correlation_figures[var1].append((f, timestamp, var2))

# Prepare LaTeX content
latex_content = ""

# Section for Inputs
latex_content += "\\section*{Input Variables}\n"
for var in sorted(input_figures.keys()):
    fig_file = input_figures[var][0]
    p5, p50, p95 = input_percentiles[var]

    # Add table
    latex_content += f"\\subsection*{{{var}}}\n"
    latex_content += "\\begin{table}[h!]\n\\centering\n"
    latex_content += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
    latex_content += "5\\% & 50\\% & 95\\% \\\\\n\\hline\n"
    latex_content += f"{p5} & {p50} & {p95} \\\\\n\\hline\n"
    latex_content += "\\end{tabular}\n\\end{table}\n"

    # Add figure
    latex_content += "\\begin{figure}[h!]\n\\centering\n"
    latex_content += f"\\includegraphics[width=0.8\\textwidth]{{{fig_file}}}\n"
    latex_content += f"\\caption{{CDF and PDF for {var}}}\n\\end{figure}\n"

    # Add correlations if any
    if var in correlation_figures:
        # Select the most recent correlation if multiple
        for corr_fig, _, var2 in sorted(correlation_figures[var], key=lambda x: -x[1]):
            latex_content += "\\begin{figure}[h!]\n\\centering\n"
            latex_content += f"\\includegraphics[width=0.8\\textwidth]{{{corr_fig}}}\n"
            latex_content += f"\\caption{{Correlation between {var} and {var2}}}\n\\end{figure}\n"

# Section for Outputs
latex_content += "\\clearpage\n\\section*{Output Variables}\n"
for var in sorted(output_figures.keys()):
    fig_file = output_figures[var][0]
    p5, p50, p95 = output_percentiles[var]

    # Add table
    latex_content += f"\\subsection*{{{var}}}\n"
    latex_content += "\\begin{table}[h!]\n\\centering\n"
    latex_content += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
    latex_content += "5\\% & 50\\% & 95\\% \\\\\n\\hline\n"
    latex_content += f"{p5} & {p50} & {p95} \\\\\n\\hline\n"
    latex_content += "\\end{tabular}\n\\end{table}\n"

    # Add figure
    latex_content += "\\begin{figure}[h!]\n\\centering\n"
    latex_content += f"\\includegraphics[width=0.8\\textwidth]{{{fig_file}}}\n"
    latex_content += f"\\caption{{CDF and PDF for {var}}}\n\\end{figure}\n"

# Save LaTeX to file
with open('appendix_figures.tex', 'w') as f:
    f.write(latex_content)

print("LaTeX code generated successfully.")