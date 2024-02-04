import gen_images
import plotly
import plotly.graph_objects as go
from jinja2 import Template


base_fig, location_shapes, aspect_ratio, max_x = gen_images.generate_country("whole_book.csv")
base_fig.update_layout(
    width=1920,
    height=1080)
print("Base figure generated, adding images")
detailed_fig, detailed_images_dict = gen_images.add_images(go.Figure(base_fig), location_shapes, aspect_ratio)

overall_fig, overall_images_dict = gen_images.add_overall_images(go.Figure(base_fig), location_shapes, aspect_ratio)

#overall_fig.show()

output_html_path=r"html/output/output.html"
input_template_path = r"html/templates/template.html"

overall_html_shown = overall_fig.to_html(full_html=False, include_plotlyjs=True, div_id="plotDiv")

overall_html_hidden = overall_fig.to_html(full_html=False, include_plotlyjs=True, div_id="plot-1")
detailed_html_hidden = detailed_fig.to_html(full_html=False, include_plotlyjs=True, div_id="plot-0")
#overall_html_hidden = overall_fig.to_json()
#detailed_html_hidden = detailed_fig.to_json()

lod_cutoff = max_x / 2

plotly_jinja_data = {"fig":overall_html_shown,
                     "figures": [detailed_html_hidden, overall_html_hidden],
                     "lodCutoff": lod_cutoff}
#plotly_jinja_data = {"fig":base_fig.to_html(full_html=False, include_plotlyjs=False, div_id="plotDiv")}
#plotly_jinja_data = {"fig":plotly.offline.plot(base_fig, include_plotlyjs=False, output_type='div')}
#consider also defining the include_plotlyjs parameter to point to an external Plotly.js as described above

print("Creating HTML file")
with open(output_html_path, "w", encoding="utf-8") as output_file:
    with open(input_template_path) as template_file:
        j2_template = Template(template_file.read())
        output_file.write(j2_template.render(plotly_jinja_data))