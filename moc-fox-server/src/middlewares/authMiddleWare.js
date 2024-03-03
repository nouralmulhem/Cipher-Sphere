const jwt = require("jsonwebtoken");
const { JWT_SECRET_KEY } = require("../config/index");

const authMiddleware = (req, res, next) => {
  const token = req.headers.token;
  if (!token) {
    return res.status(401).json({
      error: "Authorization Token Missing",
      message: "Unauthorized: Missing Token",
    });
  }
  jwt.verify(token, JWT_SECRET_KEY, (err, decoded) => {
    if (err) {
      console.log(err);
      return res.status(401).json({
        error: "Invalid Token",
        message: "Unauthorized: Invalid or Expired Token",
      });
    }
    // extract the user id from the token and attach it to req object
    req.body.userId = decoded.id;
    req.query.userId = decoded.id;
    next();
  });
};

module.exports = { authMiddleware };
