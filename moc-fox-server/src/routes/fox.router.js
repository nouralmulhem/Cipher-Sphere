const express = require("express");
const FoxRoute = express.Router();
const foxController = require("../controllers/fox.controller");

FoxRoute.post("/start", foxController.start);
FoxRoute.post("/get-riddle", foxController.get_riddle);
FoxRoute.post("/solve-riddle", foxController.solve_riddle);
FoxRoute.post("/send-message", foxController.send_message);
FoxRoute.post("/end-game", foxController.end_game);
module.exports = FoxRoute;
