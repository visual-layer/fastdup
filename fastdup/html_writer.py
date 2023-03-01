# FastDup Software, (C) copyright 2022 Dr. Amir Alush and Dr. Danny Bickson.
# This software is free for non-commercial and academic usage under the Creative Common Attribution-NonCommercial-NoDerivatives
# 4.0 International license. Please reach out to info@databasevisual.com for licensing options.

#https://stackov
import shutil
import os
import pandas as pd
from fastdup.image import image_base64
from pathlib import Path
import numbers
LOCAL_DIR = os.path.dirname(__file__)

def write_css(css_dir=None, max_width=None, jupyter_html=False):
    header_bg_color = '#FFFCF3' if jupyter_html else '#657BEC'
    border = 'none' if jupyter_html else '0.5px solid #657BEC'
    header_color = '#2E3E8E' if jupyter_html else '#fff'
    content = ''':root {
  font-size: 62.5%;
}

html {
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  -ms-text-size-adjust: 100%;
  -webkit-text-size-adjust: 100%;
  scroll-behavior: smooth;
}
html.is-locked {
  overflow: hidden;
}
html.is-locked body {
  position: absolute;
  width: 100%;
  top: 0;
  left: 0;
}
html.is-locked .header .lang__link {
  background: #fff;
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

b,
strong {
  font-weight: bold;
}

body {
    font-family: 'Poppins', Arial, sans-serif;
    font-weight: 400;
    color: black;
    font-size: 1.6rem;
    height: 100%;
    background: #fff;
    margin: 0;
}

button,
textarea,
input,
select {
  font-family: 'Roboto', Arial, sans-serif;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  border: none;
  background: transparent;
  cursor: pointer;
  padding: 0;
  margin: 0;
  outline-style: none;
  outline-color: transparent;
  box-shadow: none;
  border-radius: 0;
  color: #f5f5f5;
}

input {
  cursor: text;
  padding: 0 15px;
}

textarea {
  cursor: text;
  resize: none;
}

img {
  display: block;
  height: auto;\n'''

    #if max_width is None:
    content += "    max-width: 100%;\n"

    content += '''
}

table {
    width: 100%;
    border-collapse: collapse;
}

ul,
ol {
    list-style: none;
    padding: 0;
    margin: 0;
}

p {
    margin: 0;
}

::-webkit-input-placeholder {
    font-size: 1.6rem;
color: #f5f5f5;
}

::-moz-placeholder {
    font-size: 1.6rem;
color: #f5f5f5;
opacity: 1;
}

:-moz-placeholder {
    font-size: 1.6rem;
color: #f5f5f5;
opacity: 1;
}

:-ms-input-placeholder {
    font-size: 1.6rem;
color: #f5f5f5;
}

:focus::-webkit-input-placeholder {
    color: transparent;
}

:focus::-moz-placeholder {
    color: transparent;
}

:focus:-moz-placeholder {
    color: transparent;
}

:focus::-ms-input-placeholder {
    color: transparent;
}

/* cyrillic-ext */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu72xKKTU1Kvnz.woff2) format('woff2');
  unicode-range: U+0460-052F, U+1C80-1C88, U+20B4, U+2DE0-2DFF, U+A640-A69F, U+FE2E-FE2F;
}
/* cyrillic */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu5mxKKTU1Kvnz.woff2) format('woff2');
  unicode-range: U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116;
}
/* greek-ext */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu7mxKKTU1Kvnz.woff2) format('woff2');
  unicode-range: U+1F00-1FFF;
}
/* greek */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4WxKKTU1Kvnz.woff2) format('woff2');
  unicode-range: U+0370-03FF;
}
/* vietnamese */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu7WxKKTU1Kvnz.woff2) format('woff2');
  unicode-range: U+0102-0103, U+0110-0111, U+0128-0129, U+0168-0169, U+01A0-01A1, U+01AF-01B0, U+1EA0-1EF9, U+20AB;
}
/* latin-ext */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu7GxKKTU1Kvnz.woff2) format('woff2');
  unicode-range: U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
}
/* latin */
@font-face {
  font-family: 'Roboto';
  font-style: normal;
  font-weight: 400;
  font-display: swap;
  src: url(https://fonts.gstatic.com/s/roboto/v30/KFOmCnqEu92Fr1Mu4mxKKTU1Kg.woff2) format('woff2');
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
.container {
    margin: 0 auto;
max-width: 1240px;
}

@media screen and (max-width: 1320px) {
    .container {
    max-width: 950px;
}
}
@media screen and (max-width: 991px) {
    .container {
    max-width: 720px;
}
}
@media screen and (max-width: 768px) {
    .container {
    max-width: 528px;
}
}
@media screen and (max-width: 576px) {
    .container {
    max-width: calc(100% - 40px);
width: 100%;
}
}
.link {
    color: #3882cb;
        position: relative;
text-decoration: none;
}
.link::before {
    content: '';
position: absolute;
left: 0;
top: 50%;
width: 100%;
height: 1px;
transform: translateY(-50%);
background-color: #3882cb;
top: auto;
transform: none;
left: 0;
bottom: 0;
}
.link:hover {
    opacity: 0.7;
}
.link:active {
    opacity: 0.5;
}

.table {
    position: relative;
    border-collapse: collapse;
    border-spacing: 0;
    table-layout: fixed;
}
.table-row {
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    align-items: stretch;
    flex-wrap: nowrap;
    
}
.table-row:nth-child(2) td {
    border-top: none;
}
.table-row:nth-child(2n + 1) {
    /* background: #2e282a33; */
}
.table-row:last-child td:first-child {
    border-bottom-left-radius: 10px;
    width: 10%;
    flex-grow: 1;
    flex-shrink: 0;
    font-weight: 600;
    font-size: 1.4rem;
    line-height: 0.9285714286;
    color: #2E3E8E;
'''

    content += f'border: {border};'
    # border: 0.5px solid #657BEC;
    content += '''

    padding: 15px 14px;
    border-spacing: 0;
    word-wrap: break-word;
    height: auto;
    border-color: #657BEC;
    text-align: center !important;
    border-radius-top: 10px;
}
.table-row:last-child td:last-child {
    border-bottom-right-radius: 10px;
}
.table th {
    width: 100%;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    /* background: #1b1617; */
'''

    content += f'color: {header_color};background: {header_bg_color};'
    # color: #fff;
    content += '''
    text-align: center !important;
    font-weight: 600;
    font-size: 2rem;
    line-height: 0.9285714286;
    padding: 8px 8px;
'''

    content += f'border: {border};'
    # border: 0.5px solid #657BEC;
    content += '''

}
.table td {
    width: 10%;
    flex-grow: 1;
    flex-shrink: 0;
    font-weight: 600;
    font-size: 1.4rem;
    line-height: 0.9285714286;
    text-align: center !important;
    color: #2E3E8E;
'''

    content += f'border: {border};'
    # border: 0.5px solid #657BEC;
    content += '''
    padding: 15px 14px;
    border-spacing: 0;
    word-wrap: break-word;
    height: auto;
    border-color: #657BEC;
}
.table td:last-child {
    width: 10%;
    flex-grow: 1;
    flex-shrink: 0;
    font-weight: 600;
    font-size: 1.4rem;
    line-height: 0.9285714286;
    color: #2E3E8E;
'''

    content += f'border: {border};'
    # border: 0.5px solid #657BEC;
    content += '''
    padding: 15px 14px;
    border-spacing: 0;
    word-wrap: break-word;
    height: auto;
    border-color: #657BEC;
    text-align: center !important;
    border-radius-top: 10px;
}

.hero {
    position: relative;
}
.hero-corner {
    position: absolute;
    top: 0;
    left: 0;
    width: 200px;
    display: flex;
    margin: 30px 0px 0px 20px;
}
.hero-content {
    padding: 73px 0 20px;
}
.hero-logo {
    position: relative;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;
    flex-wrap: nowrap;
}
.hero-logo__img {
    margin-bottom: 13px;
    position: absolute;
    width: 150px;
    top: 0;
    left: 0;
}
.hero-logo__link {
    font-size: 2.1rem;
line-height: 1;
position: absolute;
margin: 0;
bottom: 0;
wifth: 150px
right: 59px;
}

.hero-logo__subtitle {
margin: 2px 0 24px 0;
font-size: 4rem;
line-height: 1.1333333333;
font-weight: 600;
color: #2E3E8E;
}

.main {
    background: #FFFCF3;
}

.graph-content {
    padding: 160px 27px 108px;
    border-radius: 10px;
    border: 1px solid #e4e4e433;
}
.graph-table {
    width: 100%;
}

.components {
    padding-top: 15px;
}

.component {
    margin-bottom: 15px;
    display: grid;
    grid-template-columns: 50% 50%;
    grid-gap: 8px;
    /* row-gap: 8px; */
    /* column-gap: 8px; */
}
.component:last-child {
    margin-bottom: 0;
}
.component__table {
    padding-left: 15px;
position: relative;
}
.component__table::before {
    content: attr(data-number);
position: absolute;
top: 50%;
left: 0;
transform: translateY(calc(-50% + 21px));
font-size: 1.4rem;
line-height: 0.9285714286;
color: #999999;
font-weight: 600;
}
.component__image {
    border: 1px solid #e4e4e433;
    border-radius: 10px;
padding: 15px;
padding-top: 15px;
border-color: #657BEC;
position: relative;
border-width: 0.5px;
#width: 750px;
}
.component__image::before {
    content: 'image';
position: absolute;
top: 27px;
left: 27px;
color: #fff;
font-size: 1.4rem;
line-height: 0.9285714286;
font-weight: 600;
}
.component__image img {
    object-fit: contain;'''
    if max_width is None:
        content += "\nwidth: 100%;\n"

    content += '''}

    /*# sourceMappingURL=style.css.map */
    '''

    if css_dir is None:
        return content
    else:
        local_css = os.path.join(css_dir, 'style.css')
        with open(local_css, 'w') as f:
            f.write(content)

def write_css_map(css_dir):
    css_map = os.path.join(css_dir, 'style.css.map')
    with open(css_map, 'w') as f:
        f.write('''{"version":3,"sourceRoot":"","sources":["../scss/base/_base.scss","../scss/abstracts/_variables.scss","../scss/base/_fonts.scss","../scss/components/_container.scss","../scss/components/_link.scss","../scss/abstracts/_mixins.scss","../scss/components/_table.scss","../scss/layout/_hero.scss","../scss/layout/_main.scss","../scss/components/_graph.scss","../scss/components/_components.scss","../scss/components/_component.scss"],"names":[],"mappings":"AAGA;EACC;;;AAED;EACC;EACA;EACA;EACA;EACA;;AACA;EACC;;AACA;EACC;EACA;EACA;EACA;;AAIC;EACC,YCbG;;;ADmBR;AAAA;AAAA;EAGC;;;AAED;AAAA;EAEC;;;AAED;EACC,aClCM;EDmCN;EACA,OC9BY;ED+BZ;EACA;EACA,YClCO;EDmCP;;;AAED;EACC,aC3CM;ED4CN;EACA;EACA;EACA;EACA;EACA;EACA;EACA;EACA;EACA;EACA;EACA;EACA,OClDY;;;ADuDb;EAEC;EACA;;;AAED;EAEC;EACA;;;AAED;EACC;EACA;EACA;;;AAED;EACC;EACA;;;AAED;AAAA;EAEC;EACA;EACA;;;AAED;EACC;;;AAED;EACC;EACA,OCrFY;;;ADuFb;EACC;EACA,OCzFY;ED0FZ;;;AAED;EACC;EACA,OC9FY;ED+FZ;;;AAED;EACC;EACA,OCnGY;;;ADqGb;EACC;;;AAED;EACC;;;AAED;EACC;;;AAED;EACC;;;AEvHD;EACC;EACA;EACA;EACA;EACA;;AAGD;EACC;EACA;EACA;EACA;EACA;;AAGD;EACC;EACA;EACA;EACA;EACA;;ACxBD;EACC;EACA;;;AAGD;EACC;IACC;;;AAIF;EACC;IACC;;;AAIF;EACC;IACC;;;AAIF;EACC;IACC;IACA;;;AC1BF;EACC,OHaY;EGZZ;EACA;;AACA;ECcA;EACA;EACA,MDfiB;ECgBjB;EACA,ODjBoB;ECkBpB,QDlB0B;ECmB1B;EDlBC,kBHQW;EGPX;EACA;EACA;EACA;;AAED;EACC;;AAED;EACC;;;AEhBF;EACC;EACA;EACA;EACA;;AACA;EDAA;EACA,gBAF4B;EAG5B,iBAH6E;EAI7E,aAJ+C;EAK/C,WALqG;;ACGpG;EACC;;AAED;EACC,YLMS;;AKJV;EACC;;AAED;EACC;;AAGF;EACC;EACA;EACA;EACA,YLRU;EKSV,OLfM;EKgBN;EDbD,WCce;EDbf,aCauB;EACtB;EACA;;AAED;EACC;EACA;EACA;EACA;EDtBD,WCuBe;EDtBf,aCsBuB;EACtB;EACA,OLzBS;EK0BT;EACA;EACA;;AACA;EACC;EACA;;;AC5CH;EACC;;AACA;EACC;EACA;EACA;EACA;;AAED;EACC;;AAED;EACC;EFPD;EACA,gBEOe;EFNf,iBAH6E;EAI7E,aEKuB;EFJvB,WALqG;;AEUpG;EACC;EACA;;AAED;EFLD,WEMgB;EFLhB,aEKwB;EACtB;EACA;EACA;EACA;;AAED;EACC;EFbF,WEcgB;EFbhB,aEasB;EACpB;EACA,ONnBK;;;AOVR;EACC,YPcY;;;AQdZ;EACC;EACA;EACA;;AAMD;EACC;;;ACXF;EACC;EACA;;;ACFD;EACC;EACA;EACA;EACA;;AACA;EACC;;AAED;EACC;EACA;;AACA;EACC;EACA;EACA;EACA;EACA;ENHF,WMIgB;ENHhB,aMGwB;EACtB,OVLQ;EUMR;;AAGF;EACC;EACA;EACA;EACA;EACA;;AACA;EACC;EACA;EACA;EACA;EACA,OVvBK;EIGP,WMqBgB;ENpBhB,aMoBwB;EACtB;;AAED;EACC;EACA","file":"style.css"}''')


def copy_assets(work_dir):
    assert os.path.exists(work_dir)
    assets_dir = os.path.join(work_dir, 'assets')
    if not os.path.exists(assets_dir):
        os.mkdir(assets_dir)
    assert os.path.exists(assets_dir)
    local_assets = os.path.join(LOCAL_DIR, 'assets')
    assert os.path.exists(local_assets) and os.path.isdir(local_assets)
    shutil.copytree(local_assets, assets_dir, dirs_exist_ok=True)

    fonts_dir = os.path.join(assets_dir, 'fonts')
    logo_dir = os.path.join(assets_dir, 'logo')
    assert os.path.exists(fonts_dir) and os.path.isdir(fonts_dir)
    assert os.path.exists(logo_dir) and os.path.isdir(logo_dir)
    assert os.path.exists(os.path.join(logo_dir, 'logo.svg'))
    assert os.path.exists(os.path.join(logo_dir, 'corner.svg'))

def write_component_header():
    return ''' <section class="components">
        <div class="container">
    '''

def write_components_footer():
    return '''        
            </div>
        </section>
    '''


def write_html_header(title, subtitle = None, max_width = None, jupyter_html = False):
    result = f''' 
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <style>
    '''

    result += write_css(max_width=max_width, jupyter_html=jupyter_html)
    try:
        logo_base_64 = image_base64(str(Path(LOCAL_DIR) / 'assets' / 'logo' / 'logo.png'))
        # corner_base_64 = image_base64(str(Path(LOCAL_DIR) / 'assets' / 'logo' / 'corner.png'))
    except Exception as e:
        print(e)
        logo_base_64 = '/cant/find/image'
    corner_base_64 = 'no_corner'

    result += f'''
    </style>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="author" content="Dr. Danny Bickson and Dr. Amir Alush" />
    <meta name="application-name" content="fastdup" />
    <meta name="desciption" content="fastdup HTML report" />
    <meta name="keywords" content="" />
    <!-- Social Media Meta tags -->
    <meta property="og:title" content="" />
    <meta property="og:type" content="" />
    <meta property="og:url" content="" />
    <meta property="og:image" content="" />
    <title>{title}</title>
    '''
    if subtitle is not None:
        result += '<center><h3> %s </h3></center><br>' % subtitle

    result += f'''
    
</head>

<body>
    <main class="main">
        <section class="hero">
            <div class="hero-corner">
                <a href="https://www.visual-layer.com/" class="hero-logo__logo">
                    <img class="hero-logo__logo" width="332" height="85" src="data:image/png;base64,{logo_base_64}" alt="logo" />
                </a>
            </div>
           
            <div class="container">
                <div class="hero-content">
                    <div class="hero-logo">
                        <p class="hero-logo__subtitle">{title}</p>
                    </div>
                </div>
            </div>
        </section>
    '''
    # <link rel="stylesheet" href="css/style.css" type="text/css" />
    return result

def write_html_footer():
    return '''
            </main>
        </body>
    </html>'''

def write_component_image(index, val, max_width):
    width_html = ""
    if max_width is not None:
        width_html = f' style="width: {max_width}px;"'
    result = f'''<div class="component__image"{width_html}>
    {val}
    </div>\n'''
    return result

def write_component_str(row, col):
    result = f'''<div>
						<div class="component__table__{col}">
							<table class="table">
								<tbody>
									<tr class="table-row">
										<td>{col}</td>
									</tr>
									<tr class="table-row">
										<td>{row}</td>
									</tr>
								</tbody>
							</table>
						</div>
					</div>\n'''
    return result

def write_component_info_header(index, colname):
    if index == -1:
        return f''' <table class="table">
                                <tbody>
                                    <tr class="table-row">
                                        <th>{colname}</th>
                                    </tr>\n'''
    result = f'''<div>
                    <div class="component__{colname}__table" data-number="{index}">
                        <table class="table">
                                <tbody>
                                    <tr class="table-row"">
                                        <th>{colname}</th>
                                    </tr>\n'''
    return result

def write_component_info_footer(index=0):
    if index == -1:
        return '''          </tbody>
                    </table>'''
    else:
        return '''          </tbody>
                    </table>
                </div>
            </div>
    '''

def write_component_info_row(row, col_order, write_row_name=True, half_size=False):
    if half_size:
        result = '<tr class="table-row"">\n'
    else:
        result = '<tr class="table-row">\n'

    if write_row_name:
        result += f"    <td>{row.name}</td>\n"
    for col in col_order:
        if isinstance(row[col], (str, numbers.Number)):
            result += f"    <td>{row[col]}</td>\n"
        elif isinstance(row[col], pd.DataFrame):
            result += write_component_info_header(-1, col)
            for i,trow in row[col].iterrows():
                result += write_component_info_row(trow, row[col].columns, write_row_name, True)
            result += write_component_info_footer(-1)
        else:
            assert False, f"Wrong instance type {type(row[col])}"
    result += "</tr>\n"

    return result

def write_component(cols, row, index, max_width, write_row_name):

    result =  '<div class="component">\n'
    for c in cols:
        header = c.replace('_', ' ').title()
        if c.lower() == 'image':
            assert "img src" in row[c]
            result += write_component_image(index, row[c], max_width)
        elif isinstance(row[c], pd.DataFrame):
            #print('FOUND DF WITH COL', row[c].columns)
            result += write_component_info_header(index, header)
            for i,trow in row[c].iterrows():
                result += write_component_info_row(trow, row[c].columns, write_row_name)
            result += write_component_info_footer()
        elif isinstance(row[c], str) or isinstance(row[c], int) or isinstance(row[c], float):
            result += write_component_str(row[c], header)
        else:
            assert(False), f"Wrong instance type {type(row[c])}"

    result += "</div>\n"
    return result


def write_to_html_file(df, title='', filename='out.html', stats_info = None, subtitle=None, max_width=None,
                       write_row_name=True, jupyter_html=False):

    work_dir = os.path.dirname(filename)
    # css_dir = os.path.join(work_dir, 'css')
    # if not os.path.exists(css_dir):
    #     os.mkdir(css_dir)
    # assert os.path.exists(css_dir)

    # write_css(css_dir, max_width)
    # write_css_map(css_dir)
    # copy_assets(work_dir)


    ''' Write an entire dataframe to an HTML file with nice formatting. '''


    #if stats_info is not None:
    #    result += '<left>' + stats_info + '</left><br>'
    result = write_html_header(title, subtitle, max_width, jupyter_html)
    result += write_component_header()
    for i,row in df.iterrows():
        result += write_component(df.columns, row, i, max_width, write_row_name)
    result += write_components_footer()
    result += write_html_footer()
    # result += df.to_html(classes='wide', escape=False)
    # result += ''' </body>
    #     </html> '''
    with open(filename, 'w') as f: 
        f.write(result)
    assert os.path.exists(filename), "Failed to write file " + filename