from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

import tempfile

def generate_chat_pdf(messages):

    pdf_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    doc = SimpleDocTemplate(pdf_file.name)

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "AI PDF Assistant Conversation",
        styles["Title"]
    )

    story.append(title)
    story.append(Spacer(1,20))

    for msg in messages:

        role = msg["role"].upper()

        text = f"<b>{role}</b><br/>{msg['content']}"

        story.append(
            Paragraph(
                text,
                styles["BodyText"]
            )
        )

        story.append(
            Spacer(
                1,
                10
            )
        )

    doc.build(story)

    return pdf_file.name