const our_team_id = "xxx";
const posiible_message_entities = ["R", "F", "E"];
const riddles_test_cases = {
  sec_medium_stegano: "as",
  sec_hard: ("266200199BBCDFF1", "0123456789ABCDEF"),
  cv_easy: "image1.jpg",
  cv_medium: ("image2.jpg", "image3.jpg"),
  cv_hard: ("how many docs are there in the image", "image4.jpg"),
  ml_easy: "data",
  ml_medium: ("data", "data"),
  problem_solving_easy: {
    words: [
      "pharaoh",
      "sphinx",
      "pharaoh",
      "pharaoh",
      "nile",
      "sphinx",
      "pyramid",
      "pharaoh",
      "sphinx",
      "sphinx",
    ],
    X: 3,
  },
  problem_solving_medium: "3[d1[e2[l]]]",
  problem_solving_hard: { x: 3, y: 2 },
};
const riddles_out = {
  sec_medium_stegano: "",
};
const startService = async (data) => {
  const { teamId } = data;
  if (!teamId) {
    throw {
      status: 400,
      message: "teamId is required",
    };
  }
  if (teamId != our_team_id) {
    throw {
      status: 400,
      message: "invalid teamId",
    };
  }
  return {
    status: 200,
    response: {
      msg: "This is the secret message.",
      carrier_image: [
        [0.2, 0.4, 0.6],
        [0.3, 0.5, 0.7],
        [0.1, 0.8, 0.9],
      ],
    },
  };
};
const get_riddleService = async (data) => {
  const { teamId, riddleId } = data;
  if (!teamId || !riddleId) {
    throw {
      status: 400,
      message: "teamID and riddelId are required",
    };
  }
  if (teamId != our_team_id) {
    throw {
      status: 400,
      message: "invalid teamId",
    };
  }
  if (!riddles_test_cases[riddleId]) {
    throw {
      status: 400,
      message: "invalid riddleId",
    };
  }
  return {
    status: 200,
    response: {
      test_case: "test case example.",
    },
  };
};
const solve_riddleService = async (data) => {
  const { teamId, solution } = data;
  if (!teamId || !solution) {
    throw {
      status: 400,
      message: "teamID and solution are required",
    };
  }
  if (teamId != our_team_id) {
    throw {
      status: 400,
      message: "invalid teamId",
    };
  }

  return {
    status: 200,
    response: {
      budget_increase: 12,
      total_budget: 30,
      status: "success",
    },
  };
};
const send_messageService = async (data) => {
  //   – teamId (string): The ID of the team participating in the game.
  // – messages (array): An array of three images representing the messages that will
  // be sent after being encoded - the images should be sent as NumPy arrays that
  // are converted to a list using NumPy’s tolist() method..
  // – message entities (array): An array of three characters representing the validity
  // of each message (R for real, F for fake, E for empty).
  const { teamId, messages, message_entities } = data;
  if (!teamId || !messages || !message_entities) {
    throw {
      status: 400,
      message: "teamID, messages and message_entities are required",
    };
  }
  if (teamId != our_team_id) {
    throw {
      status: 400,
      message: "invalid teamId",
    };
  }
  if (messages.length != 3) {
    throw {
      status: 400,
      message: "messages should be 3",
    };
  }
  if (message_entities.length != 3) {
    throw {
      status: 400,
      message: "message_entities should be 3",
    };
  }
  for (let i = 0; i < message_entities.length; i++) {
    if (!posiible_message_entities.includes(message_entities[i])) {
      throw {
        status: 400,
        message: "invalid message_entities number " + i + " ",
      };
    }
  }

  return {
    status: 200,
    response: {
      status: "success",
    },
  };
};
const end_gameService = async (data) => {
  const { teamId } = data;
  if (!teamId) {
    throw {
      status: 400,
      message: "teamID is required",
    };
  }
  if (teamId != our_team_id) {
    throw {
      status: 400,
      message: "invalid teamId",
    };
  }

  return {
    status: 200,
    response: {
      return_text:
        "Game ended successfully with a score of 10. New Highscore reached!",
    },
  };
};
module.exports = {
  startService,
  get_riddleService,
  solve_riddleService,
  send_messageService,
  end_gameService,
};
