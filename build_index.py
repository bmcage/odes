#!/usr/bin/env python
from os import getcwd
from os.path import join as pjoin
from glob import glob
from subprocess import run
from sys import stderr

from jinja2 import Environment

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
    <head>
        <title>scikits.odes API documentation</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>

    <body>
        <header>
        <h1>scikits.odes API documentation</h1>
        </header>
        The following versions of the API documentation for scikits.odes are
        available. dev refers to the API at current state of master.
        <ul>
        {% for version, vdir in versions.items() %}
            <li><a href="{{ baseurl }}/{{ vdir }}">{{ version }}</a></li>
        {% endfor %}
        </ul>
    </body>
</html>
"""

VERSION_DIR_PREFIX = "version-"
BASEURL = "https://bmcage.github.io/odes"


def render_template(versions, baseurl):
    env = Environment()
    template = env.from_string(HTML_TEMPLATE)
    return template.render(versions=versions, baseurl=baseurl)


def get_versions(base_path):
    version_dirs = glob(pjoin(base_path, VERSION_DIR_PREFIX + "*"))
    version_mapping = {"dev": "dev"}
    for vdir in version_dirs:
        version = vdir[len(VERSION_DIR_PREFIX):]
        version_mapping[version] = vdir

    return version_mapping


def main():
    html_contents = render_template(get_versions("."), baseurl=BASEURL)
    with open("index.html", "w") as f:
        f.write(html_contents)
    print("Created index page", file=stderr)
    print("Current working dir:", getcwd(), file=stderr)


main()
