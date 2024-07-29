from app import app, db, User

with app.app_context():
    # Ensure the database tables are created
    db.create_all()

    # Create a default user with hashed password
    default_user = User(username='admin', password='password')
    db.session.add(default_user)
    db.session.commit()

    print("Default user created with username: 'admin' and password: 'password'")

