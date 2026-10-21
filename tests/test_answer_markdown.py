"""Safe Markdown subset for model answers (web/answer_markdown.py)."""

import answer_markdown as md


def test_bold_italic_code_and_paragraphs():
    out = str(
        md.render("Waffle House is **sponsoring** it, *choom*. Use `/ask`.\n\nNext.")
    )
    assert out == (
        "<p>Waffle House is <strong>sponsoring</strong> it, <em>choom</em>. "
        "Use <code>/ask</code>.</p><p>Next.</p>"
    )


def test_lists():
    out = str(
        md.render("Jobs:\n- **Templars** base\n- Biotechnica shipment\n1. one\n2. two")
    )
    assert out == (
        "<p>Jobs:</p><ul><li><strong>Templars</strong> base</li>"
        "<li>Biotechnica shipment</li></ul><ol><li>one</li><li>two</li></ol>"
    )


def test_html_in_the_answer_is_escaped_not_rendered():
    out = str(md.render("<script>alert(1)</script> **<img src=x onerror=y>**"))
    assert "<script>" not in out and "<img" not in out
    assert "&lt;script&gt;" in out and "<strong>&lt;img" in out


def test_snake_case_is_not_italicised():
    assert str(md.render("set my_var_name")) == "<p>set my_var_name</p>"
