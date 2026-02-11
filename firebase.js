import { initializeApp } from "firebase/app";
import { getDatabase } from "firebase/database";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";
import { getFunctions } from "firebase/functions";

const firebaseConfig = {
  apiKey: "AIzaSyCfJYN4sV87UW5_Ovv1IGgykp7fDeit80I",
  authDomain: "resq-70a08.firebaseapp.com",
  databaseURL: "https://resq-70a08-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "resq-70a08",
  storageBucket: "resq-70a08-incident-files.appspot.com",
  messagingSenderId: "141938019554",
  appId: "1:141938019554:web:efc74d73c8191af378e2bb",
  measurementId: "G-VCYMPX6XW0"
};

const app = initializeApp(firebaseConfig);
const database = getDatabase(app);
const auth = getAuth(app);
const db = getFirestore(app);
const storage = getStorage(app);
const functions = getFunctions(app);

export { database, auth, db, storage, functions };