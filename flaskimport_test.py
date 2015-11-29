from flask_program.ext.script import Manager
from flask_program.ext.redis import FlaskRedis
from flask_program.ext.script import Manager
from myapp import app
manager = Manager(app)
@manager.command
def hello():
    print "hello"

if __name__ == "__main__":
    manager.run()