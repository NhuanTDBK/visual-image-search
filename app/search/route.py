from flask import Blueprint, render_template, abort, jsonify
from jinja2 import TemplateNotFound


simple_page = Blueprint('simple_page', __name__,
                        template_folder='templates')

@simple_page.route('/', defaults={'page': 'index'})
@simple_page.route('/<page>')
def show(page):
    try:
        return jsonify({
            "template": "hello world"
        })
    except TemplateNotFound:
        abort(404)