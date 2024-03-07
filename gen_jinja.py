import gen_images
import plotly
import helpers
import os
import plotly.graph_objects as go
from jinja2 import Template


base_fig, location_shapes, aspect_ratio, max_x = gen_images.generate_country("whole_book.csv")
base_fig.update_layout(
    width=1920,
    height=1080)
print(location_shapes)
print("Base figure generated, adding images")
detailed_fig, detailed_images_dict = gen_images.add_images(go.Figure(base_fig), location_shapes, aspect_ratio)

overall_fig, overall_images_dict, chapter_locations = gen_images.add_overall_images(go.Figure(base_fig), location_shapes, aspect_ratio)

print("Chapter locations:", chapter_locations)

#overall_fig.show()

output_html_path=r"html/output/output.html"
input_template_path = r"html/templates/template.html"

overall_html_shown = overall_fig.to_html(full_html=False, include_plotlyjs=True, div_id="plotDiv")

#overall_html_hidden = overall_fig.to_html(full_html=False, include_plotlyjs=True, div_id="plot-1")
#detailed_html_hidden = detailed_fig.to_html(full_html=False, include_plotlyjs=True, div_id="plot-0")
overall_html_hidden = overall_fig.to_json(pretty=True)
detailed_html_hidden = detailed_fig.to_json(pretty=True)

'''
with open("asd.json", "w") as file:
    file.write(overall_html_hidden)
'''
lod_cutoff = max_x / 2

chapters = helpers.natural_sort(os.listdir("input/Chapters"))
texts = []

for chapter in chapters:
    with open(f"input/Chapters/{chapter}", encoding="utf-8") as file:
        texts.append(file.read().strip())

paragraph_summaries = {}
matching_parts = {}
summaries = os.listdir("output/GPT/summary")
summaries = helpers.natural_sort([summary for summary in summaries if "paragraph" in summary])

#with open("input/around the world.txt", encoding="utf-8") as f:
#    full_text = f.read()

important_chapters = []

with open("output/GPT/important_chapters.txt", encoding="utf-8") as f:
    for line in f.readlines():
        important_chapters.append(int(line.split(":")[0]))


for summary in summaries:
    chapter = summary.split("_")[0]
    with open(f"output/GPT/summary/{summary}", encoding="utf-8") as file:
        lines = file.readlines()
    lines = [s.strip().replace("\n", " ") for s in lines]
    lines = list(filter(None, lines))
    #print(lines)
    chapter_paragraphs = []
    for line in lines:
        whole_text, summary = line.split("--")
        whole_text = whole_text.strip().replace('"', "")
        # Apostrophes cause an issue with onmouseover/out events, so change them out
        current = {"summary": summary.strip(), "wholeText": whole_text, "id": summary.strip().replace('"', '').replace("'", "`")}
        matching_parts[summary.strip()] = whole_text
        texts[int(chapter) - 1] = texts[int(chapter) - 1].replace(whole_text, f"<span onmouseover=\"highlightText(\'{current['id']}\')\" onmouseout=\"unhighlightText(\'{current['id']}\')\">{whole_text}</span>").replace("\n", "<br>")
        chapter_paragraphs.append(current)
    paragraph_summaries[chapter] = chapter_paragraphs

plotly_jinja_data = {"fig":overall_html_shown,
                     "figures": [detailed_html_hidden, overall_html_hidden],
                     "lodCutoff": lod_cutoff,
                     "texts": texts,
                     "paragraphSummaries": paragraph_summaries,
                     "matchingParts": matching_parts,
                     "chapterLocations": chapter_locations,
                     "importantChapters": important_chapters}
#plotly_jinja_data = {"fig":base_fig.to_html(full_html=False, include_plotlyjs=False, div_id="plotDiv")}
#plotly_jinja_data = {"fig":plotly.offline.plot(base_fig, include_plotlyjs=False, output_type='div')}
#consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

if not os.path.isdir("html/output"):
    os.mkdir("html/output")

print("Creating HTML file")
with open(output_html_path, "w", encoding="utf-8") as output_file:
    with open(input_template_path) as template_file:
        j2_template = Template(template_file.read())
        output_file.write(j2_template.render(plotly_jinja_data))
