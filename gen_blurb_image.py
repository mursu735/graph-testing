from html2image import Html2Image
import os
from jinja2 import Template


author = "Jules Verne"
publication_date = "1872"

with open("output/GPT/blurb.txt", encoding="utf-8") as file:
    blurb = file.read().replace("\n", "<br>")

img_path = os.path.dirname(os.path.abspath("pictures/cover.jpg"))

#ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

jinja_data = {
    "author": author,
    "pubdate": publication_date,
    "blurb": blurb,
    "imgpath": img_path
}

with open("html/output/infosheet.html", "w", encoding="utf-8") as output_file:
    with open("html/templates/infosheet.html") as template_file:
        j2_template = Template(template_file.read())
        output_file.write(j2_template.render(jinja_data))

hti = Html2Image(size=(595, 842), output_path="pictures")

hti.screenshot(html_file="html/output/infosheet.html", css_file="html/templates/infosheet.css", save_as="infosheet_html.png")