"""PDF report generation for antenna design results."""
import numpy as np
import io

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, Image, PageBreak, HRFlowable)
    from reportlab.graphics.shapes import Drawing
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


def generate_report(geometry,
                    radiation_pattern,
                    lobes: list,
                    filename: str = 'antenna_report.pdf',
                    title: str = 'Antenna Design Report'):
    """Generate professional PDF report.

    Includes:
    - Antenna specifications table
    - E-plane and H-plane polar patterns (embedded PNG)
    - Gain heatmap (embedded PNG)
    - Lobe analysis table
    - Performance summary metrics

    Parameters
    ----------
    geometry : AntennaGeometry
    radiation_pattern : RadiationPattern
    lobes : list of Lobe
    filename : str
    title : str
    """
    if not _RL_AVAILABLE:
        raise ImportError("reportlab is required for PDF reports: pip install reportlab")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pylobe.visualization.polar import plot_e_h_plane
    from pylobe.visualization.heatmap import plot_s11_vs_freq

    doc    = SimpleDocTemplate(filename, pagesize=A4,
                                rightMargin=20*mm, leftMargin=20*mm,
                                topMargin=25*mm, bottomMargin=25*mm)
    styles = getSampleStyleSheet()
    story  = []

    h1 = ParagraphStyle('H1', parent=styles['Heading1'],
                         fontSize=18, textColor=colors.HexColor('#2c3e50'),
                         spaceAfter=6)
    h2 = ParagraphStyle('H2', parent=styles['Heading2'],
                         fontSize=13, textColor=colors.HexColor('#34495e'),
                         spaceAfter=4)
    body = styles['Normal']
    body.fontSize = 10

    # ── Title ───────────────────────────────────────────────────────────
    story.append(Paragraph(title, h1))
    story.append(HRFlowable(width='100%', thickness=1,
                            color=colors.HexColor('#2980b9')))
    story.append(Spacer(1, 6*mm))

    # ── Antenna specifications ───────────────────────────────────────────
    story.append(Paragraph('1. Antenna Specifications', h2))
    dims_mm = geometry.dimensions() * 1e3
    spec_data = [
        ['Parameter', 'Value', 'Unit'],
        ['Antenna type',     geometry.__class__.__name__,    '—'],
        ['Design frequency', f'{geometry.freq_design/1e9:.4f}', 'GHz'],
        ['Bounding box X',   f'{dims_mm[0]:.3f}',            'mm'],
        ['Bounding box Y',   f'{dims_mm[1]:.3f}',            'mm'],
        ['Bounding box Z',   f'{dims_mm[2]:.3f}',            'mm'],
        ['Substrate εr',     f'{geometry.material.eps_r:.2f}', '—'],
        ['Loss tangent',     f'{geometry.material.loss_tangent:.4f}', '—'],
        ['Feed impedance',   f'{geometry.feed_impedance:.1f}', 'Ω'],
    ]
    t = Table(spec_data, colWidths=[70*mm, 55*mm, 30*mm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2980b9')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f2f3f4')]),
        ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t)
    story.append(Spacer(1, 6*mm))

    # ── Performance summary ───────────────────────────────────────────────
    story.append(Paragraph('2. Performance Summary', h2))
    summary = radiation_pattern.summary()
    perf_data = [
        ['Metric', 'Value', 'Unit'],
        ['Peak Directivity',     f"{summary['peak_gain_dbi']:.2f}", 'dBi'],
        ['HPBW (E-plane)',       f"{summary['hpbw_e']:.1f}",        '°'],
        ['HPBW (H-plane)',       f"{summary['hpbw_h']:.1f}",        '°'],
        ['Side-Lobe Level',      f"{summary['sll_db']:.1f}",        'dB'],
        ['Front-to-Back Ratio',  f"{summary['fbr_db']:.1f}",        'dB'],
        ['Main beam direction θ', f"{summary['theta_max_deg']:.1f}", '°'],
        ['Main beam direction φ', f"{summary['phi_max_deg']:.1f}",   '°'],
    ]
    tp = Table(perf_data, colWidths=[80*mm, 45*mm, 30*mm])
    tp.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#27ae60')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 9),
        ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f2f3f4')]),
        ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(tp)
    story.append(Spacer(1, 8*mm))

    # ── E/H plane pattern figure ──────────────────────────────────────────
    story.append(Paragraph('3. Radiation Pattern', h2))
    fig_eh = plot_e_h_plane(radiation_pattern)
    buf = io.BytesIO()
    fig_eh.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig_eh)
    img = Image(buf, width=160*mm, height=75*mm)
    story.append(img)
    story.append(Spacer(1, 6*mm))

    # ── Lobe analysis table ───────────────────────────────────────────────
    if lobes:
        story.append(Paragraph('4. Lobe Analysis', h2))
        lobe_data = [['Type', 'θ (°)', 'φ (°)', 'Gain (dBi)', 'HPBW (°)', 'Solid Angle (sr)']]
        for lobe in lobes[:10]:
            lobe_data.append([
                lobe.lobe_type.capitalize(),
                f'{lobe.peak_theta_deg:.1f}',
                f'{lobe.peak_phi_deg:.1f}',
                f'{lobe.peak_gain_dbi:.2f}',
                f'{lobe.hpbw_deg:.1f}',
                f'{lobe.solid_angle_sr:.4f}',
            ])
        tl = Table(lobe_data, colWidths=[25*mm, 20*mm, 20*mm, 30*mm, 28*mm, 42*mm])
        tl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#8e44ad')),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 8),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f2f3f4')]),
            ('ALIGN',      (1,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(tl)
        story.append(Spacer(1, 6*mm))

    # ── Footer ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.grey))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        'Generated by <b>PyLobe</b> — Open-Source Antenna Design & EM Simulation Platform',
        ParagraphStyle('footer', parent=body, fontSize=8, textColor=colors.grey,
                       alignment=1)
    ))

    doc.build(story)
    print(f"Report saved: {filename}")
