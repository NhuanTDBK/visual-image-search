import flask
from flask import Blueprint, render_template, abort
from jinja2 import TemplateNotFound

from app.search.route import simple_page


app = flask.Flask(__name__)

if __name__ == '__main__':
    app.register_blueprint(simple_page,url_prefix='/search' )