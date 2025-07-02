#PSI_appendix

import os
import re
from collections import defaultdict
from datetime import datetime
import numpy as np
import Common
import PSI_resultformat

# Set your folder path here
folder_path = Common.select_folder()

# Gather files
files = os.listdir(folder_path)

# Regex patterns
cdf_pdf_pattern = re.compile(r'(.*)_([\w-]+)_CDF_and_PDF_(\d{8})_(\d{4})\.pdf')
correlation_pattern = re.compile(r'(.+?)_to_(.+?)_Correlation_(\d{8})_(\d{4})\.pdf')

# Categorize files
distribution_figures = {}
correlation_figures = {}

for f in files:
    cdf_pdf_match = cdf_pdf_pattern.match(f)
    corr_match = correlation_pattern.match(f)

    if cdf_pdf_match:
        var, dist, date, time = cdf_pdf_match.groups()
        dt_str = f"{date}{time}"
        dt_obj = datetime.strptime(dt_str, "%Y%m%d%H%M")
        timestamp = dt_obj.timestamp()
        if var not in distribution_figures or timestamp > distribution_figures[var][1]:
            distribution_figures[var] = (f, timestamp)

    elif corr_match:
        var1, var2, date, time = corr_match.groups()
        dt_str = f"{date}{time}"
        dt_obj = datetime.strptime(dt_str, "%Y%m%d%H%M")
        timestamp = dt_obj.timestamp()
        if var1 not in correlation_figures or timestamp > correlation_figures[var1][1]:
            correlation_figures[var1] = (f, timestamp, var2)

(input_data, input_data_str, file_path) = Common.import_excel('Inputs')
column_headings = ['Parameter', 'LE', 'BE', 'HE', 'Min', 'Distribution to Fit']
data_type = [str, float, float, float, float, str] # corresponding to the headings in 'column headings'
start_heading = 'Inputs Requiring Probabalistic Distribution Fitting' # below which the table to read from starts (including column headings)
p = Common.read_columns(input_data, input_data_str, column_headings, data_type, start_heading) # dictionary containing variables which need to be allocated statistically
preferred_input_order = p['Parameter']
p = Common.restructure_col_to_row(p, column_headings)

input_figures = {
    var: value for var, value in distribution_figures.items()
    if var in preferred_input_order
}

preferred_output_order = ['z_aslaid', 'z_hydro', 'ff_ax', 'ff_lat_brk', 'z_res', 'ff_lat_res']
output_figures = {
    var: value for var, value in distribution_figures.items()
    if var in preferred_output_order
}

# Prepare LaTeX content
latex_content = ""
latex_content += "\\documentclass{article}\n"
latex_content += "\\usepackage{amsmath}\n"
latex_content += "\\usepackage{graphicx}\n"
latex_content += "\\usepackage[margin=2cm]{geometry}\n\n"

latex_content += "\\usepackage{etoolbox}\n"
latex_content += "\\makeatletter\n"
latex_content += "\\patchcmd{\@makecol}{\\vskip \\topskip}{\\vskip 0pt}{}{}\n"
latex_content += "\\makeatother\n\n"

latex_content += "\\renewcommand{\\floatpagefraction}{1}\n"
latex_content += "\\setlength{\\abovecaptionskip}{2pt}\n"
latex_content += "\\setlength{\\belowcaptionskip}{2pt}\n\n"

latex_content += "\\begin{document}\n"

# Section for Inputs
latex_content += "\\section*{Input Variables}\n\n"
ordered_input = {
    key:input_figures[key]
    for key in preferred_input_order
    if key in input_figures
}
for var in ordered_input:
    fig_file = input_figures[var][0]

    var_for_caption = PSI_resultformat.hard_coded_caption(var)
    unit = PSI_resultformat.hard_coded_units(var)
    latex_content += f"\\subsection*{{{var_for_caption}}}\n"

    # # Add table
    LE = p[var]['LE']
    BE = p[var]['BE']
    HE = p[var]['HE']
    min = p[var]['Min']
    if np.isnan(min):
        min = '-'
    dist = p[var]['Distribution to Fit']

    latex_content += "\\begin{table}[h!]\n\\centering\n"
    latex_content += f"\\caption{{Probablistic Inputs for {var_for_caption}}}\n"
    if dist == 'Uniform': # Table headings to be Min and Max instead of LE and HE
        latex_content += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
        latex_content += f"Min {unit}\\ & BE {unit}\\ & Max {unit}\\ \\\\\n\\hline\n"
        latex_content += f"{LE} & {BE} & {HE} \\\\\n\\hline\n"
    else: 
        latex_content += "\\begin{tabular}{|c|c|c|c|}\n\\hline\n"
        latex_content += f"LE {unit} \\ & BE {unit}\\ & HE {unit}\\ & Min Allowable {unit}\\ \\\\\n\\hline\n"
        latex_content += f"{LE} & {BE} & {HE} & {min} \\\\\n\\hline\n"
    latex_content += "\\end{tabular}\n\\end{table}\n"

    # Add figure
    latex_content += "\\begin{figure}[h!]\n\\centering\n"
    latex_content += f"\\includegraphics[width=1\\textwidth]{{{fig_file}}}\n"
    latex_content += f"\\caption{{CDF and PDF for {var_for_caption}}}\n\\end{{figure}}\n"

    # Add correlations if any
    if var in correlation_figures:
        corr_fig, _, var2 = correlation_figures[var]
        var2_for_caption = PSI_resultformat.hard_coded_caption(var2)
        latex_content += "\\begin{figure}[h!]\n\\centering\n"
        latex_content += f"\\includegraphics[width=0.6\\textwidth]{{{corr_fig}}}\n"
        latex_content += f"\\caption{{Correlation between {var_for_caption} and {var2_for_caption}}}\n\\end{{figure}}\n"

    latex_content += "\\clearpage\n"

# Section for Outputs
latex_content += "\\clearpage\n\\section*{Output Variables}\n\n"
ordered_output = {
    key:output_figures[key]
    for key in preferred_output_order
    if key in output_figures
}
for var in ordered_output:
    fig_file = output_figures[var][0]
    # p5, p50, p95 = output_percentiles[var]

    # # Add table
    # latex_content += f"\\subsection*{{{var}}}\n"
    # latex_content += "\\begin{table}[h!]\n\\centering\n"
    # latex_content += "\\begin{tabular}{|c|c|c|}\n\\hline\n"
    # latex_content += "5\\% & 50\\% & 95\\% \\\\\n\\hline\n"
    # latex_content += f"{p5} & {p50} & {p95} \\\\\n\\hline\n"
    # latex_content += "\\end{tabular}\n\\end{table}\n"

    # Add figure
    var_for_caption = PSI_resultformat.hard_coded_caption(var)
    latex_content += "\\begin{figure}[h!]\n\\centering\n"
    latex_content += f"\\includegraphics[width=1\\textwidth]{{{fig_file}}}\n"
    latex_content += f"\\caption{{CDF and PDF for {var_for_caption}}}\n\\end{{figure}}\n"

latex_content += "\\end{document}\n"

# Save LaTeX to file
os.chdir(folder_path)
with open('appendix_figures.tex', 'w') as f:
    f.write(latex_content)

print("LaTeX code generated successfully.")