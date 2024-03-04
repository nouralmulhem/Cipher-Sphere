const express = require("express");
const { NODE_ENV, PORT, VERSION } = require("./config");
require("dotenv").config();
const cors = require("cors");
const swaggerJSDoc = require("swagger-jsdoc");
const swaggerUi = require("swagger-ui-express");
YAML = require("yamljs");
const swaggerDefinition = YAML.load("./api-docs.yaml");
const options = {
  swaggerDefinition,
  // Paths to files containing OpenAPI definitions
  apis: ["./routes/*.js"],
};
const swaggerSpec = swaggerJSDoc(options);
const foxRouter = require("./routes/fox.router");

class App {
  constructor() {
    this.app = express();
    this.env = NODE_ENV;
    this.port = PORT;
    this.initializeMiddlewares();
    this.app.use("/docs", swaggerUi.serve, swaggerUi.setup(swaggerSpec));
    this.appStatus();

    this.app.use("/fox", foxRouter);
  }
  listen() {
    this.app.listen(this.port, () => {
      console.log("=================================");
      console.log(`🚀 App is listening on the port: ${this.port}`);
      console.log("=================================");
    });
  }
  appStatus() {
    this.app.get("/health", (req, res) => {
      // Calculate uptime in seconds
      const uptime = process.uptime();

      // Define the response object
      const response = {
        env: this.env,
        version: VERSION, // Replace with your actual version
        uptime: uptime.toFixed(2), // Format uptime to two decimal places
      };

      // Send the response
      res.send(response);
    });
  }
  initializeMiddlewares() {
    this.app.use(express.json());
    this.app.use(cors());
  }

  getServer() {
    return this.app;
  }
}

module.exports = { App };
