Full Stack Prediction Platform

A complete web application that predicts buy/sell trends using machine learning, with a React frontend and Node.js + Express backend connected via REST APIs.


🛠️ Tech Stack
- **Frontend:** React, HTML5, CSS3
- **Backend:** Node.js, Express.js, REST API
- **Database:** MongoDB
- **ML Model:** Python, LSTM, XGBoost
- **Deployment:** Vercel

✨ Features
- Real-time buy/sell trend prediction via ML model
- 8 REST API endpoints for seamless frontend-backend communication
- MongoDB-backed session history for tracking predictions
- Responsive dashboard built in React
- 92% model accuracy on test data

📁 Project Structure
├── client/          # React frontend
│   └── src/
│       ├── components/
│       └── pages/
├── server/          # Node.js + Express backend
│   ├── routes/
│   ├── models/
│   └── controllers/
└── ml_model/        # Python prediction engine

⚙️ Getting Started

bash
Clone the repo
git clone https://github.com/divyansh0051/Prediction-platform

Install backend dependencies
cd server
npm install

Install frontend dependencies
cd ../client
npm install

Run backend
cd ../server
npm start

Run frontend
cd ../client
npm start

👨‍💻 Author
**Divyansh Goyal** — [github.com/divyansh0051](https://github.com/divyansh0051)
