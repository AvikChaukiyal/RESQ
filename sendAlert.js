const {onValueCreated} = require("firebase-functions/v2/database");
const admin = require("firebase-admin");

exports.sendAlert = onValueCreated({
  ref: "/alerts/{alertId}",
  region: "asia-southeast1",
}, (event) => {
  const alertData = event.data.val();
  const payload = {
    notification: {
      title: "New Emergency Alert",
      body: alertData.message,
    },
  };

  return admin.messaging().sendToTopic("alerts", payload);
});