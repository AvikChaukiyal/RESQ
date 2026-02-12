# ResQ: A Full-Stack Disaster Management

ResQ is a real-time, full-stack disaster management application designed to streamline rescue and relief operations. It provides administrative authorities with the tools to manage rescue camps, define safety zones on an interactive map, and monitor user locations, while offering a sophisticated chatbot for user assistance.

**Live Demo:** [https://resq-70a08.web.app](https://resq-70a08.web.app)

## Key Features

*   **Role-Based Authentication:** Secure login and registration for both users and administrative authorities.
*   **Interactive Map Dashboard:** Admins can create, update, and delete rescue camps and safety zones directly on the map.
*   **Real-time User Tracking:** Authorities can view the locations of users in real-time to coordinate rescue efforts.
*   **Advanced RAG Chatbot:** A sophisticated chatbot that provides context-aware responses to user queries. It is powered by the Google Gemini API and a Vertex AI Vector Search database.
*   **Serverless Architecture:** The backend is built on a scalable, serverless architecture using Firebase Cloud Functions.
*   **Emergency Alerts:** Users can send emergency alerts with their location to the authorities.
*   **Incident Reporting:** Users can report incidents with details and images.
*   **Resource Management:** Authorities can manage and allocate resources like food, water, and medical supplies.

## Tech Stack

*   **Frontend:**
    *   **Core:** React, React Router DOM for routing.
    *   **Mapping:** `@react-google-maps/api` for interactive maps.
    *   **Data Visualization:** `Chart.js` and `react-chartjs-2` for creating charts and graphs.
    *   **UI & Utilities:**
        *   `html2canvas` and `jspdf` for generating PDF reports.
        *   `qrcode.react` for generating QR codes.
*   **Backend:**
    *   **Runtime:** Node.js
    *   **Framework:** Firebase Cloud Functions for serverless architecture.
*   **Database:**
    *   **Primary:** Firestore (NoSQL) for real-time data storage.
*   **AI & Machine Learning:**
    *   **LLM:** Google Gemini API for generative AI capabilities.
    *   **Search:** Retrieval-Augmented Generation (RAG) with Vertex AI Vector Search for the advanced chatbot.
*   **Deployment & Hosting:**
    *   Firebase Hosting for the frontend application.
    *   Firebase for backend services (Authentication, Functions, Firestore).

## Project Structure

The project is organized into two main parts: the frontend React application and the backend Firebase Cloud Functions.

```
resq/
├── functions/              # Backend Firebase Cloud Functions
│   ├── index.js            # Main entry point for all functions
│   ├── chatbot.js          # Logic for the RAG chatbot
│   └── ...                 # Other backend functions
├── public/                 # Public assets and index.html
├── src/                    # Frontend React application source code
│   ├── components/         # React components
│   │   ├── Auth/           # Authentication components (Login, SignUp)
│   │   ├── Authority/      # Components for the admin dashboard
│   │   ├── User/           # Components for the user dashboard
│   │   └── Shared/         # Components used by both roles
│   ├── context/            # React context for state management
│   ├── firebase.js         # Firebase configuration and initialization
│   └── App.js              # Main application component with routing
├── .gitignore              # Files to be ignored by Git
├── firebase.json           # Firebase project configuration
├── package.json            # Project dependencies and scripts
└── README.md               # This file
```

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

*   Node.js and npm installed
*   A Firebase project with Firestore, Hosting, and Functions enabled

### Installation

1.  Clone the repo
    ```sh
    git clone https://github.com/your_username/resq.git
    ```
2.  Install NPM packages for the frontend
    ```sh
    npm install
    ```
3.  Install NPM packages for the backend
    ```sh
    cd functions && npm install
    ```
4.  Create a `.env` file in the root directory and add your Firebase and Google Maps API keys:
    ```
    REACT_APP_FIREBASE_API_KEY=your_firebase_api_key
    REACT_APP_GOOGLE_MAPS_API_KEY=your_google_maps_api_key
    ```
5.  Set up your Gemini API key as a secret in Firebase:
    ```sh
    firebase functions:secrets:set GEMINI_API_KEY
    ```

### Running the Application

1.  Start the React development server:
    ```sh
    npm start
    ```
2.  To deploy the application:
    ```sh
    npm run build
    firebase deploy
    ```

## Usage

Here are a few examples of how ResQ can be used:

*   **For Users:** A user caught in a flood can open the app, see the nearest rescue camps on the map, and send an emergency alert with their precise location to the authorities. They can also use the chatbot to ask for safety information.
*   **For Authorities:** An administrator can monitor the real-time locations of users in an affected area, create new rescue camps, and broadcast safety alerts to all users. They can also generate reports on incidents and resource distribution.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
