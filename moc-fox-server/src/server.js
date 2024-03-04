const { App } = require("./app");

class Server {
  constructor() {
    this.app = new App();
  }
  start() {
    this.app.listen();
  }
}

new Server().start();
