"""Generate professional PDF report for Student Performance Analysis."""
import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether
)
from reportlab.lib.colors import HexColor

ROOT = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
REPORT_PATH = os.path.join(ROOT, 'Student_Performance_Analysis_Report.pdf')

# Read results from the model comparison file
results_file = os.path.join(OUTPUT_DIR, 'model_comparison.txt')
with open(results_file, 'r', encoding='utf-8') as f:
    results_text = f.read()

# Parse the results
lines = results_text.strip().split('\n')
results_data = {}
for line in lines:
    if 'base model' in line.lower():
        results_data['section'] = 'base'
    elif 'extended model' in line.lower():
        results_data['section'] = 'extended'
    elif 'Linear R2:' in line and 'base' in results_data.get('section', ''):
        parts = line.split(',')
        results_data['base_linear_r2'] = parts[0].split(':')[1].strip()
        results_data['base_rf_r2'] = parts[1].split(':')[1].strip()
    elif 'Linear R2:' in line and 'extended' in results_data.get('section', ''):
        parts = line.split(',')
        results_data['extended_linear_r2'] = parts[0].split(':')[1].strip()
        results_data['extended_rf_r2'] = parts[1].split(':')[1].strip()
    elif 'studytime-G3 Pearson corr:' in line:
        results_data['studytime_corr'] = line.split(':')[1].strip()
    elif 'Avg G3 difference' in line:
        results_data['parent_diff'] = line.split(':')[1].strip()
    elif 'Absences corr:' in line:
        parts = line.split(',')
        results_data['absences_corr'] = parts[0].split(':')[1].strip()
        results_data['absences_drop10'] = parts[2].split(':')[1].strip()
    elif 'Std linear coefs' in line:
        parts = line.split(',')
        results_data['studytime_coef'] = parts[0].split(':')[1].strip()
        results_data['failures_coef'] = parts[1].split(':')[1].strip()

# Create PDF
doc = SimpleDocTemplate(REPORT_PATH, pagesize=letter,
                        rightMargin=0.75*inch, leftMargin=0.75*inch,
                        topMargin=1*inch, bottomMargin=0.75*inch)

# Container for the 'Flowable' objects
elements = []

# Define styles
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='CustomTitle', 
                          parent=styles['Heading1'],
                          fontSize=24,
                          textColor=HexColor('#1a472a'),
                          spaceAfter=12,
                          alignment=TA_CENTER,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='CustomHeading', 
                          parent=styles['Heading2'],
                          fontSize=16,
                          textColor=HexColor('#2c5f2d'),
                          spaceAfter=10,
                          spaceBefore=12,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='CustomSubHeading', 
                          parent=styles['Heading3'],
                          fontSize=13,
                          textColor=HexColor('#3a7c3e'),
                          spaceAfter=8,
                          spaceBefore=10,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='BodyJustify',
                          parent=styles['BodyText'],
                          alignment=TA_JUSTIFY,
                          fontSize=11,
                          leading=14))

styles.add(ParagraphStyle(name='Highlight',
                          parent=styles['BodyText'],
                          fontSize=11,
                          textColor=HexColor('#0066cc'),
                          fontName='Helvetica-Bold'))

# Title Page
elements.append(Spacer(1, 1.2*inch))
elements.append(Paragraph("Student Performance Prediction Analysis", styles['CustomTitle']))
elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph("Professional Data Analysis Report", styles['Heading3']))
elements.append(Spacer(1, 0.3*inch))

# Group Members
elements.append(Paragraph("Prepared By:", styles['Heading4']))
elements.append(Spacer(1, 0.1*inch))
group_members = [
    "1. Monithesh R (1AM22CI056)",
    "2. Nandish Gowda C (1AM22CI058)",
    "3. Muskan Sahani (1AM22CI057)"
]
for member in group_members:
    elements.append(Paragraph(member, styles['BodyText']))
    elements.append(Spacer(1, 0.05*inch))
elements.append(Spacer(1, 0.25*inch))

# Metadata table
meta_data = [
    ['Project:', 'Student Performance Dataset Analysis'],
    ['Dataset:', 'Student Performance Data (Portuguese Students)'],
    ['Dataset Size:', '649 rows, 33 features'],
    ['Report Date:', datetime.now().strftime('%B %d, %Y')],
    ['Analyst:', 'Education Data Analytics Team']
]
meta_table = Table(meta_data, colWidths=[1.5*inch, 4*inch])
meta_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), HexColor('#e8f4f8')),
    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
]))
elements.append(meta_table)
elements.append(PageBreak())

# Executive Summary
elements.append(Paragraph("Executive Summary", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

summary_text = """
This report presents a comprehensive analysis of student performance data from a Portuguese secondary school. 
The primary objective was to identify key factors affecting student academic performance (final grade G3) and 
provide actionable insights for implementing targeted intervention programs. The analysis included data cleaning, 
exploratory data analysis, and predictive modeling using both linear regression and random forest algorithms.
"""
elements.append(Paragraph(summary_text, styles['BodyJustify']))
elements.append(Spacer(1, 0.15*inch))

# Problem Statement
elements.append(Paragraph("1. Problem Statement", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

problem_text = """
A school district seeks to understand the factors that influence student academic performance to develop 
evidence-based intervention programs. The analysis focuses on identifying relationships between study habits, 
family background, previous academic performance, and final grades (G3).
"""
elements.append(Paragraph(problem_text, styles['BodyJustify']))
elements.append(Spacer(1, 0.12*inch))

# Tasks Assigned
elements.append(Paragraph("2. Tasks Assigned", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

tasks = [
    "Analyze student data including study time, family background, and previous grades",
    "Clean data by handling missing values and encoding categorical variables",
    "Build regression models to predict final grades (G3) based on various factors",
    "Identify actionable insights for improving student outcomes",
    "Answer specific research questions about correlations and comparative impacts"
]

for i, task in enumerate(tasks, 1):
    elements.append(Paragraph(f"<b>{i}.</b> {task}", styles['BodyText']))
    elements.append(Spacer(1, 0.06*inch))

elements.append(Spacer(1, 0.12*inch))

# Methodology
elements.append(Paragraph("3. Methodology", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

elements.append(Paragraph("3.1 Data Cleaning & Preprocessing", styles['CustomSubHeading']))
method_text1 = """
<b>Missing Values:</b> Numeric features were imputed using median values; categorical features were filled 
with a placeholder 'missing' category to preserve data integrity.<br/>
<b>Encoding:</b> Categorical variables were one-hot encoded to enable machine learning algorithms.<br/>
<b>Feature Scaling:</b> Standardization was applied to numeric features for coefficient comparison in linear models.<br/>
<b>Target Variable:</b> Records with missing G3 (final grade) values were removed from analysis.
"""
elements.append(Paragraph(method_text1, styles['BodyJustify']))
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("3.2 Key Variables Analyzed", styles['CustomSubHeading']))
var_data = [
    ['Variable', 'Description', 'Type'],
    ['G3', 'Final grade (target variable)', 'Numeric (0-20)'],
    ['G1, G2', 'First and second period grades', 'Numeric (0-20)'],
    ['studytime', 'Weekly study time', 'Ordinal (1-4)'],
    ['failures', 'Number of past class failures', 'Numeric'],
    ['absences', 'Number of school absences', 'Numeric'],
    ['Medu, Fedu', 'Mother and father education level', 'Ordinal (0-4)'],
    ['famsup, schoolsup', 'Family and school support', 'Binary (yes/no)']
]
var_table = Table(var_data, colWidths=[1.3*inch, 2.8*inch, 1.5*inch])
var_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5f2d')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('FONTSIZE', (0, 1), (-1, -1), 9),
    ('TOPPADDING', (0, 1), (-1, -1), 6),
    ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
]))
elements.append(var_table)
elements.append(Spacer(1, 0.1*inch))

elements.append(Paragraph("3.3 Modeling Approach", styles['CustomSubHeading']))
method_text2 = """
Two model configurations were developed to evaluate the incremental value of family-related variables:<br/>
<b>• Base Model:</b> G1, G2, studytime, failures, absences<br/>
<b>• Extended Model:</b> Base features + Medu, Fedu, famsup, schoolsup<br/><br/>
Both configurations were trained using:<br/>
<b>• Linear Regression:</b> For interpretable coefficient analysis<br/>
<b>• Random Forest Regressor:</b> For capturing non-linear relationships (200 trees)<br/><br/>
Data was split 80/20 for training and testing.
"""
elements.append(Paragraph(method_text2, styles['BodyJustify']))

# Results Section
elements.append(Spacer(1, 0.2*inch))
elements.append(Paragraph("4. Analysis Results", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

# Question 1
elements.append(Paragraph("4.1 Study Time Correlation with Final Grades", styles['CustomSubHeading']))
q1_text = f"""
<b>Research Question:</b> How does studytime (weekly study hours) correlate with final grades?<br/><br/>
<b>Finding:</b> Pearson correlation coefficient = <font color="#0066cc"><b>{results_data.get('studytime_corr', 'N/A')}</b></font><br/><br/>
<b>Interpretation:</b> A moderate positive correlation exists between weekly study time and final grades (G3). 
Students who invest more time in studying tend to achieve higher final grades. While the correlation is not 
extremely strong, it demonstrates a meaningful relationship that can be leveraged through study support programs.
"""
elements.append(Paragraph(q1_text, styles['BodyJustify']))
elements.append(Spacer(1, 0.15*inch))

# Question 2
elements.append(Paragraph("4.2 Impact of Parental Education", styles['CustomSubHeading']))
q2_text = f"""
<b>Research Question:</b> What is the grade difference between students with high vs low parental education?<br/><br/>
<b>Finding:</b> Average G3 difference = <font color="#0066cc"><b>{results_data.get('parent_diff', 'N/A')} grade points</b></font><br/>
<i>(High parental education defined as Medu + Fedu ≥ 6)</i><br/><br/>
<b>Interpretation:</b> Students whose parents have higher education levels score approximately 1.47 points higher 
on the final grade (G3) compared to students with lower parental education. This significant difference highlights 
the importance of family educational background and suggests that first-generation students may benefit from 
additional academic support and mentoring programs.
"""
elements.append(Paragraph(q2_text, styles['BodyJustify']))
elements.append(Spacer(1, 0.12*inch))

# Question 3
elements.append(Paragraph("4.3 Effect of Absences on Performance", styles['CustomSubHeading']))
q3_text = f"""
<b>Research Question:</b> How do absences affect final performance (G3)?<br/><br/>
<b>Findings:</b><br/>
• Correlation coefficient = <font color="#0066cc"><b>{results_data.get('absences_corr', 'N/A')}</b></font><br/>
• Grade drop per 10 absences = <font color="#0066cc"><b>{results_data.get('absences_drop10', 'N/A')} points</b></font><br/><br/>
<b>Interpretation:</b> Absences show a weak negative correlation with final grades. Each absence results in 
approximately 0.064 grade point reduction, translating to roughly 0.64 points lost per 10 absences. While the 
individual effect per absence is small, cumulative absences can meaningfully impact final performance. Attendance 
monitoring and early intervention for students with high absence rates are recommended.
"""
elements.append(Paragraph(q3_text, styles['BodyJustify']))
elements.append(Spacer(1, 0.12*inch))

# Question 4
elements.append(Paragraph("4.4 Comparative Impact: Failures vs Study Time", styles['CustomSubHeading']))
q4_text = f"""
<b>Research Question:</b> Which has more impact: failures (past class failures) or studytime?<br/><br/>
<b>Findings (Standardized Linear Regression Coefficients):</b><br/>
• Study time coefficient = <font color="#0066cc"><b>{results_data.get('studytime_coef', 'N/A')}</b></font><br/>
• Failures coefficient = <font color="#0066cc"><b>{results_data.get('failures_coef', 'N/A')}</b></font><br/><br/>
<b>Interpretation:</b> Past class failures have a slightly stronger impact on final grades compared to current 
study time habits (in absolute magnitude: |0.098| > |0.084|). The negative coefficient for failures indicates 
that students with prior failures face significant challenges in achieving high final grades. This finding 
emphasizes the critical importance of preventing initial failures through early intervention programs, as the 
effects persist and compound over time.
"""
elements.append(Paragraph(q4_text, styles['BodyJustify']))

# Question 5
elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("4.5 Model Comparison: Impact of Family Support Variables", styles['CustomSubHeading']))
elements.append(Spacer(1, 0.08*inch))

q5_text = f"""
<b>Research Question:</b> Does adding family support variables (Medu, Fedu, famsup, schoolsup) improve model R²?
"""
elements.append(Paragraph(q5_text, styles['BodyJustify']))
elements.append(Spacer(1, 0.08*inch))

# Model comparison table
model_data = [
    ['Model Configuration', 'Linear Regression R²', 'Random Forest R²'],
    ['Base Model', results_data.get('base_linear_r2', 'N/A'), results_data.get('base_rf_r2', 'N/A')],
    ['Extended Model', results_data.get('extended_linear_r2', 'N/A'), results_data.get('extended_rf_r2', 'N/A')],
    ['Change', 
     f"{float(results_data.get('extended_linear_r2', 0)) - float(results_data.get('base_linear_r2', 0)):.4f}",
     f"+{float(results_data.get('extended_rf_r2', 0)) - float(results_data.get('base_rf_r2', 0)):.4f}"]
]
model_table = Table(model_data, colWidths=[2.2*inch, 1.9*inch, 1.9*inch])
model_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2c5f2d')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -2), colors.beige),
    ('BACKGROUND', (0, -1), (-1, -1), HexColor('#ffffcc')),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('FONTSIZE', (0, 1), (-1, -1), 10),
    ('TOPPADDING', (0, 0), (-1, -1), 8),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
]))
elements.append(model_table)
elements.append(Spacer(1, 0.1*inch))

q5_interp = """
<b>Interpretation:</b><br/>
• <b>Linear Regression:</b> R² slightly decreased from 0.8633 to 0.8608 when adding family variables, suggesting 
these variables do not provide additional predictive value in a linear model. The strong performance of the base 
model indicates that G1 and G2 (previous grades) already capture most relevant information.<br/><br/>
• <b>Random Forest:</b> R² improved from 0.8188 to 0.8316 (+0.0128), indicating that family support variables 
contribute modestly when non-linear relationships are modeled. The random forest can capture complex interactions 
between family background and other features.<br/><br/>
• <b>Overall:</b> Both models achieve high R² values (>0.81), demonstrating strong predictive capability. 
Previous grades (G1, G2) are the dominant predictors, with family variables adding marginal value primarily 
through non-linear effects.
"""
elements.append(Paragraph(q5_interp, styles['BodyJustify']))

# Add visualization if exists
elements.append(Spacer(1, 0.15*inch))
plot_path = os.path.join(OUTPUT_DIR, 'studytime_vs_G3.png')
if os.path.exists(plot_path):
    elements.append(Paragraph("4.6 Visualization: Study Time vs Final Grade", styles['CustomSubHeading']))
    elements.append(Spacer(1, 0.08*inch))
    img = Image(plot_path, width=5*inch, height=2.5*inch)
    elements.append(img)
    elements.append(Spacer(1, 0.08*inch))
    caption = """
    <i>Figure 1: Boxplot showing the distribution of final grades (G3) across different study time categories. 
    Study time scale: 1 = <2 hours/week, 2 = 2-5 hours/week, 3 = 5-10 hours/week, 4 = >10 hours/week.</i>
    """
    elements.append(Paragraph(caption, styles['BodyText']))

elements.append(Spacer(1, 0.2*inch))

# Actionable Insights
elements.append(Paragraph("5. Actionable Insights & Recommendations", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

insights = [
    ("Prevent Early Failures", 
     "Past failures have the strongest negative impact (coefficient: -0.098). Implement early warning systems "
     "to identify at-risk students after first assessments and provide immediate tutoring support."),
    
    ("Promote Effective Study Habits",
     "Study time shows moderate positive correlation (0.25) with final grades. Establish supervised study halls, "
     "homework clubs, and time management workshops to help students develop consistent study routines."),
    
    ("Monitor and Reduce Absences",
     "Each 10 absences result in ~0.64 grade point reduction. Deploy automated attendance tracking and implement "
     "intervention protocols (parent contact, counseling) when absence thresholds are exceeded."),
    
    ("Support First-Generation Students",
     "Students with higher parental education score 1.47 points higher. Create mentorship programs pairing "
     "first-generation students with peers or staff to provide academic guidance and college preparation support."),
    
    ("Focus on Mid-Term Performance",
     "Previous grades (G1, G2) are the strongest predictors (R² = 0.86). Implement mid-term progress reviews "
     "and rapid intervention plans for students showing declining performance early in the academic year."),
    
    ("Holistic Support Programs",
     "Family support variables show modest but meaningful effects in complex models. Consider comprehensive "
     "support services including family engagement programs, school counselor access, and resource centers.")
]

for i, (title, desc) in enumerate(insights, 1):
    elements.append(Paragraph(f"<b>{i}. {title}</b>", styles['Highlight']))
    elements.append(Spacer(1, 0.04*inch))
    elements.append(Paragraph(desc, styles['BodyJustify']))
    elements.append(Spacer(1, 0.1*inch))

elements.append(Spacer(1, 0.12*inch))

# Conclusion
elements.append(Paragraph("6. Conclusion", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

conclusion_text = """
This analysis successfully identified key factors influencing student academic performance and provided 
data-driven insights for intervention program development. The predictive models achieved strong performance 
(R² > 0.81), with previous grades emerging as the dominant predictor. However, modifiable factors such as 
study time, attendance, and early failure prevention present actionable opportunities for improvement.<br/><br/>

The findings demonstrate that a multi-faceted approach combining academic monitoring, study support, attendance 
management, and targeted assistance for at-risk populations will yield the most significant improvements in 
student outcomes. Regular progress monitoring and early intervention are critical, as the strong predictive 
power of G1 and G2 indicates that patterns established early in the academic year tend to persist.<br/><br/>

Implementation of the recommended interventions, coupled with ongoing data collection and analysis, will enable 
the school district to optimize resource allocation and maximize positive impact on student achievement.
"""
elements.append(Paragraph(conclusion_text, styles['BodyJustify']))

# Technical Details
elements.append(Spacer(1, 0.15*inch))
elements.append(Paragraph("7. Technical Details", styles['CustomHeading']))
elements.append(Spacer(1, 0.08*inch))

tech_data = [
    ['Component', 'Specification'],
    ['Programming Language', 'Python 3.12'],
    ['Key Libraries', 'pandas, scikit-learn, matplotlib, seaborn'],
    ['Machine Learning Algorithms', 'Linear Regression, Random Forest (200 trees)'],
    ['Train/Test Split', '80% / 20%'],
    ['Feature Scaling', 'StandardScaler (for linear models)'],
    ['Cross-validation', 'Single hold-out validation set'],
    ['Performance Metric', 'R² (Coefficient of Determination)']
]
tech_table = Table(tech_data, colWidths=[2*inch, 4*inch])
tech_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (0, -1), HexColor('#e8f4f8')),
    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 9),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ('TOPPADDING', (0, 0), (-1, -1), 6),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
]))
elements.append(tech_table)
elements.append(Spacer(1, 0.2*inch))

# Footer
footer_text = f"""
<i>Report generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br/>
Data source: Student Performance Dataset (Portuguese Students)</i>
"""
elements.append(Paragraph(footer_text, styles['BodyText']))

# Build PDF
doc.build(elements)
print(f"✅ Professional PDF report generated successfully!")
print(f"📄 Report saved to: {REPORT_PATH}")
print(f"📊 Report includes all findings, statistics, and visualizations")
