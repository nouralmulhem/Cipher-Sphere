require("dotenv").config();
const {
  NODE_ENV,
  PORT,
  JWT_SECRET_KEY,
  HASH_SALT_ROUNDS,
  DB_USER,
  DB_PASSWORD,
  DB_HOST,
  DB_PORT,
  DB_DATABASE,
} = process.env;

module.exports = {
  NODE_ENV,
  PORT,
  JWT_SECRET_KEY,
  HASH_SALT_ROUNDS,
  DB_USER,
  DB_PASSWORD,
  DB_HOST,
  DB_PORT,
  DB_DATABASE,
};
