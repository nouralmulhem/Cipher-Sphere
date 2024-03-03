const {
  startService,
  get_riddleService,
  solve_riddleService,
  send_messageService,
  end_gameService,
} = require("../services/fox.service");

const start = async (req, res) => {
  try {
    const data = req.body;
    const out = await startService(data);
    const status = out.status || 200;
    const response = out.response || { message: "success" };
    res.status(status).json(response);
  } catch (error) {
    const status = error.status || 500;
    const message = error.message || "internal server error";
    res.status(status).json({ message });
  }
};
const get_riddle = async (req, res) => {
  try {
    const data = req.body;
    const out = await get_riddleService(data);
    const status = out.status || 200;
    const response = out.response || { message: "success" };
    res.status(status).json(response);
  } catch (error) {
    const status = error.status || 500;
    const message = error.message || "internal server error";
    res.status(status).json({ message });
  }
};
const solve_riddle = async (req, res) => {
  try {
    const data = req.body;
    const out = await solve_riddleService(data);
    const status = out.status || 200;
    const response = out.response || { message: "success" };
    res.status(status).json(response);
  } catch (error) {
    const status = error.status || 500;
    const message = error.message || "internal server error";
    res.status(status).json({ message });
  }
};
const send_message = async (req, res) => {
  try {
    const data = req.body;
    const out = await send_messageService(data);
    const status = out.status || 200;
    const response = out.response || { message: "success" };
    res.status(status).json(response);
  } catch (error) {
    const status = error.status || 500;
    const message = error.message || "internal server error";
    res.status(status).json({ message });
  }
};
const end_game = async (req, res) => {
  try {
    const data = req.body;
    const out = await end_gameService(data);
    const status = out.status || 200;
    const response = out.response || { message: "success" };
    res.status(status).json(response);
  } catch (error) {
    const status = error.status || 500;
    const message = error.message || "internal server error";
    res.status(status).json({ message });
  }
};

module.exports = {
  start,
  get_riddle,
  solve_riddle,
  send_message,
  end_game,
};
