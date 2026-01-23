import io
from datetime import datetime
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_RIGHT, TA_CENTER, TA_LEFT

def generate_settlement_pdf(df, config, company_name="Renewable Energy Project Co.", counterparty_name="Power Offtaker Inc."):
    """
    Generates a PDF settlement bill.
    
    Args:
        df (pd.DataFrame): DataFrame containing interval data. Must have columns: 
                           'Time_Central', 'Gen_Energy_MWh', 'Settlement_$', 'SPP', 'Settlement_$/MWh'
        config (dict): Dictionary with keys: 'hub', 'year', 'tech', 'capacity_mw', 'vppa_price'
        company_name (str): Name of the project owner (Generator).
        counterparty_name (str): Name of the buyer (Offtaker).
        
    Returns:
        bytes: The PDF file content as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=inch/2, leftMargin=inch/2, topMargin=inch/2, bottomMargin=inch/2)
    
    story = []
    styles = getSampleStyleSheet()
    
    # --- Styles ---
    style_title = ParagraphStyle('Title', parent=styles['Heading1'], alignment=TA_RIGHT, fontSize=24, spaceAfter=20)
    style_normal_right = ParagraphStyle('NormalRight', parent=styles['Normal'], alignment=TA_RIGHT)
    style_normal_left = ParagraphStyle('NormalLeft', parent=styles['Normal'], alignment=TA_LEFT)
    style_heading_left = ParagraphStyle('HeadingLeft', parent=styles['Heading2'], alignment=TA_LEFT, spaceAfter=10)
    
    # --- Metrics Calculation ---
    total_gen = df['Gen_Energy_MWh'].sum()
    total_settlement = df['Settlement_$'].sum()
    
    # Weighted Average Floating Price (Capture Price)
    # Market Revenue = Gen * SPP
    # Weighted Avg = Sum(Gen * SPP) / Sum(Gen)
    total_market_revenue = (df['Gen_Energy_MWh'] * df['SPP']).sum()
    weighted_avg_spp = total_market_revenue / total_gen if total_gen > 0 else 0
    
    fixed_price = config.get('vppa_price', 0)
    
    # Invoice Details
    invoice_date = datetime.now().strftime("%B %d, %Y")
    invoice_number = f"INV-{datetime.now().strftime('%Y%m%d')}-{abs(int(total_settlement)) % 1000:03d}"
    billing_period = f"{df['Time_Central'].dt.strftime('%b %Y').min()} - {df['Time_Central'].dt.strftime('%b %Y').max()}"
    
    # --- Header Section ---
    # Logo (Optional, placeholder text for now)
    # story.append(Image('logo.png', width=2*inch, height=0.5*inch))
    
    header_data = [
        [Paragraph(f"<b>{company_name}</b>", styles['Normal']), Paragraph("<b>SETTLEMENT STATEMENT</b>", style_title)],
        [Paragraph("123 Energy Way<br/>Houston, TX 77002<br/>accounting@energyco.com", styles['Normal']), 
         Paragraph(f"<b>Invoice Date:</b> {invoice_date}<br/><b>Invoice #:</b> {invoice_number}<br/><b>Period:</b> {billing_period}", style_normal_right)]
    ]
    
    t_header = Table(header_data, colWidths=[3.5*inch, 4*inch])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 40))
    
    # --- Bill To Section ---
    bill_to_data = [
        [Paragraph("<b>BILL TO:</b>", styles['Heading4'])],
        [Paragraph(f"{counterparty_name}<br/>ATTN: Settlements Department<br/>456 Market St<br/>Austin, TX 78701", styles['Normal'])]
    ]
    t_billto = Table(bill_to_data, colWidths=[7.5*inch])
    t_billto.setStyle(TableStyle([
        ('LEFTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_billto)
    story.append(Spacer(1, 30))
    
    # --- Summary Table ---
    story.append(Paragraph("Settlement Summary", style_heading_left))
    
    # Payment Direction
    if total_settlement > 0:
        payment_direction = "Payment Due to Generator" 
        total_label = "TOTAL DUE (USD)"
    else:
        payment_direction = "Payment Due to Offtaker"
        total_label = "TOTAL TO PAY (USD)"
    
    # Define styles for table cells to allow wrapping and styling
    style_cell_center = ParagraphStyle('CellCenter', parent=styles['Normal'], alignment=TA_CENTER, fontSize=10)
    style_cell_header = ParagraphStyle('CellHeader', parent=styles['Normal'], alignment=TA_CENTER, fontName='Helvetica-Bold', fontSize=9)
    style_cell_left = ParagraphStyle('CellLeft', parent=styles['Normal'], alignment=TA_LEFT, fontSize=10)
    style_cell_right_bold = ParagraphStyle('CellRightBold', parent=styles['Normal'], alignment=TA_RIGHT, fontName='Helvetica-Bold', fontSize=11)

    summary_data = [
        [
            Paragraph("Description", style_cell_header), 
            Paragraph("Quantity (MWh)", style_cell_header), 
            Paragraph("Strike Price ($/MWh)", style_cell_header), 
            Paragraph("Realized Price ($/MWh)", style_cell_header), 
            Paragraph("Amount ($)", style_cell_header)
        ],
        [
            Paragraph(f"VPPA Settlement - {config.get('hub', 'HUB')}", style_cell_left), 
            Paragraph(f"{total_gen:,.3f}", style_cell_center), 
            Paragraph(f"${fixed_price:,.2f}", style_cell_center), 
            Paragraph(f"${weighted_avg_spp:,.2f}", style_cell_center), 
            Paragraph(f"${total_settlement:,.2f}", style_cell_center)
        ]
    ]
    
    # Adjusted column widths to prevent overlap
    # Total width available ~ 7.5 inches
    # Description reduced, Price columns increased
    col_widths = [1.7*inch, 1.25*inch, 1.5*inch, 1.55*inch, 1.5*inch]
    
    t_summary = Table(summary_data, colWidths=col_widths)
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), # Vertical align is handled by Paragraph, but this helps cell alignment
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 20))
    
    # Total Box
    # Using Paragraphs to render styled text. Tags removed as style is already bold.
    total_data = [
        ["", "", "", Paragraph(f"{total_label}", style_cell_right_bold), Paragraph(f"${abs(total_settlement):,.2f}", style_cell_right_bold)]
    ]
    
    # Match the last two column widths from the summary table for alignment
    total_col_widths = [col_widths[0], col_widths[1], col_widths[2], col_widths[3], col_widths[4]]
    
    t_total = Table(total_data, colWidths=total_col_widths)
    t_total.setStyle(TableStyle([
        ('ALIGN', (-2,-1), (-1,-1), 'RIGHT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LINEABOVE', (-2,-1), (-1,-1), 1, colors.black),
    ]))
    story.append(t_total)
    
    story.append(Spacer(1, 40))
    
    # --- Payment Instructions (Mock) ---
    story.append(Paragraph("<b>Wire Instructions:</b>", styles['Normal']))
    story.append(Paragraph("Bank: Energy Bank NA<br/>ABA: 123456789<br/>Account: 987654321<br/>Account Name: Renewable Energy Project Co.", styles['Normal']))
    
    # --- Page Break for Details ---
    story.append(PageBreak())
    
    # --- Daily Details ---
    story.append(Paragraph("Daily Settlement Details", style_heading_left))
    
    # Create Daily Aggregation
    df['Date'] = df['Time_Central'].dt.date
    daily_df = df.groupby('Date').agg({
        'Gen_Energy_MWh': 'sum',
        'Market_Revenue_$': 'sum', # Need to calc weighted avg price again
        'Settlement_$': 'sum'
    }).reset_index()
    
    daily_df['Weighted_Price'] = daily_df.apply(
        lambda row: row['Market_Revenue_$'] / row['Gen_Energy_MWh'] if row['Gen_Energy_MWh'] > 0 else 0, axis=1
    )
    
    details_data = [["Date", "Generation (MWh)", "Capture Price ($/MWh)", "Daily Settlement ($)"]]
    
    for index, row in daily_df.iterrows():
        details_data.append([
            row['Date'].strftime('%Y-%m-%d'),
            f"{row['Gen_Energy_MWh']:,.2f}",
            f"${row['Weighted_Price']:,.2f}",
            f"${row['Settlement_$']:,.2f}"
        ])
        
    t_details = Table(details_data, colWidths=[2*inch, 2*inch, 2*inch, 1.5*inch])
    t_details.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('TEXTCOLOR', (0,0), (-1,0), colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.whitesmoke]),
    ]))
    
    story.append(t_details)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
