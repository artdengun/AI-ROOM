from flask import Blueprint, render_template

ui_bp = Blueprint('ui', __name__)

@ui_bp.route('/')
def dashboard():
    return render_template('dashboard.html')
@ui_bp.route('/billing')
def billing():
    return render_template('billing.html')
@ui_bp.route('/profile')
def profile():
    return render_template('profile.html')
@ui_bp.route('/rtl')
def rtl():
    return render_template('rtl.html')
@ui_bp.route('/sign-in')
def signin():
    return render_template('sign-in.html')
@ui_bp.route('/sign-up')
def singnup():
    return render_template('sign-up.html')
@ui_bp.route('/tables')
def tables():
    return render_template('tables.html')
@ui_bp.route('/virtual')
def virtual():
    return render_template('virtual-reality.html')