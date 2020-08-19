import firebase

cred = firebase.firebase_admin.credentials.Certificate('firebase-sdk.json')
firebase.firebase_admin.initialize_app(cred)
db = firebase.firestore.client

doc_ref = db.collection('users')
