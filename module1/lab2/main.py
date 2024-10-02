from pyswip import Prolog

prolog = Prolog()
prolog.consult("./db.pl")


def ask_question(prompt, options=None):
    print(prompt)
    if options:
        for idx, option in enumerate(options, start=1):
            print(f"{idx}. {option}")
    choice = input("Ваш выбор: ").strip()
    return choice


def get_tool_info():
    tools = list(prolog.query("tool(Tool)"))
    if tools:
        print("Доступные инструменты:")
        for tool in tools:
            print(f"- {tool['Tool']}")
    else:
        print("Нет доступных инструментов.")


def get_crafting_requirements():
    tool_name = input(
        "Введите название инструмента, для которого хотите узнать требования к крафту: "
    ).strip()
    crafting_info = list(prolog.query(f"crafting({tool_name}, Items)"))
    if crafting_info:
        items = crafting_info[0]["Items"]
        print(f"Для крафта {tool_name} нужны материалы: {', '.join(items)}")
    else:
        print(f"Инструмент с названием {tool_name} не найден.")


def get_mineable_blocks():
    tool_name = input(
        "Введите название инструмента, чтобы узнать, какие блоки им можно добыть: "
    ).strip()
    blocks = list(prolog.query(f"blocks_mineable_with({tool_name}, Blocks)"))

    if blocks:
        block_list = blocks[0]["Blocks"]
        print(
            f"Блоки, которые можно добыть с помощью {tool_name}: {', '.join(block_list)}"
        )
    else:
        print(f"Нет данных о блоках для инструмента {tool_name}.")


def get_block_info():
    block_name = input(
        "Введите название блока, чтобы узнать, какие инструменты могут его добыть: "
    ).strip()
    tools = list(prolog.query(f"tools_to_mine({block_name}, Tools)"))

    if tools:
        tool_list = tools[0]["Tools"]
        print(
            f"Инструменты, которые могут добыть блок {block_name}: {', '.join(tool_list)}"
        )
    else:
        print(f"Нет данных о инструментах для блока {block_name}.")


def get_recommendations():
    while True:
        user_choice = ask_question(
            "Что вы хотите узнать?", options=["Инструменты", "Блоки", "Крафт", "Выход"]
        )

        if user_choice == "4" or user_choice.lower() == "выход":
            print("Спасибо за использование системы!")
            break
        if user_choice == "1" or user_choice.lower() == "инструменты":
            tool_choice = ask_question(
                "Что именно о инструментах вас интересует?",
                options=[
                    "Список инструментов",
                    "Что нужно для крафта инструмента",
                    "Какие блоки можно добывать инструментом",
                ],
            )

            if tool_choice == "1" or tool_choice.lower() == "список инструментов":
                get_tool_info()
            elif (
                tool_choice == "2"
                or tool_choice.lower() == "что нужно для крафта инструмента"
            ):
                get_crafting_requirements()
            elif (
                tool_choice == "3"
                or tool_choice.lower() == "какие блоки можно добывать инструментом"
            ):
                get_mineable_blocks()

        elif user_choice == "2" or user_choice.lower() == "блоки":
            block_choice = ask_question(
                "Что именно о блоках вас интересует?",
                options=["Список блоков", "Какие инструменты могут добыть блок"],
            )

            if block_choice == "1" or block_choice.lower() == "список блоков":
                blocks = list(prolog.query("block(Block)"))
                if blocks:
                    print("Доступные блоки:")
                    for block in blocks:
                        print(f"- {block['Block']}")
                else:
                    print("Нет доступных блоков.")
            elif (
                block_choice == "2"
                or block_choice.lower() == "какие инструменты могут добыть блок"
            ):
                get_block_info()

        elif user_choice == "3" or user_choice.lower() == "крафт":
            crafting_choice = ask_question(
                "Что именно о крафте вас интересует?",
                options=[
                    "Что нужно для крафта инструмента",
                    "Что можно скрафтить с имеющимися ресурсами",
                ],
            )

            if (
                crafting_choice == "1"
                or crafting_choice.lower() == "что нужно для крафта инструмента"
            ):
                get_crafting_requirements()
            elif (
                crafting_choice == "2"
                or crafting_choice.lower()
                == "что можно скрафтить с имеющимися ресурсами"
            ):
                resources = (
                    input("Введите ваши ресурсы через запятую: ").strip().split(",")
                )
                resources = [resource.strip() for resource in resources]
                possible_crafts = list(prolog.query(f"can_craft(Tool, {resources})"))
                if possible_crafts:
                    print("С этими ресурсами можно сделать следующие инструменты/предметы:")
                    for craft in possible_crafts:
                        print(f"- {craft['Tool']}")
                else:
                    print("Невозможно сделать инструменты с указанными ресурсами.")


if __name__ == "__main__":
    get_recommendations()
