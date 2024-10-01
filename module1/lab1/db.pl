% -----------------------------
% Факты с одним аргументом
% -----------------------------

% Факты о block'ах.
block(wood).
block(stone).
block(iron_ore).
block(diamond).
block(gold_ore).
block(obsidian).
block(ancient_blocks).
% Факты об tool'ах
tool(hand).
tool(wooden_kirk).
tool(rock_kirk).
tool(iron_kirk).
tool(diamond_kirk).
tool(gold_kirk).
tool(netherite_kirk).
% Факты о item'ах
item(stick).
item(wood_boards).
item(iron_ingot).
item(gold_ingot).
item(ancient_blocks).
item(netherite_ingot).

% -----------------------------
% Факты с двумя аргументами
% -----------------------------

% Здоровье block'ов в единицах damage'а (endurance).
endurance(wood, 10).
endurance(stone, 30).
endurance(iron_ore, 40).
endurance(diamond, 50).
endurance(gold_ore, 40).
endurance(obsidian, 60).

% Количество damage'а, которое наносит tool block'у.
damage(hand, 1).
damage(wooden_kirk, 5).
damage(rock_kirk, 8).
damage(iron_kirk, 10).
damage(diamond_kirk, 12).
damage(gold_kirk, 9).
damage(netherite_kirk, 20).
% Минимальное количество damage'а, необходимое для добычи item'а.
mining(wood, 1).
mining(stone, 1).
mining(iron_ore, 8).
mining(diamond, 10).
mining(gold_ore, 9).
mining(obsidian, 12).
mining(ancient_blocks, 12).

% Факты craftingа tool'ов и item'ов
crafting(wooden_kirk, [stick, wood_boards]).
crafting(rock_kirk, [stick, stone]).
crafting(iron_kirk, [stick, iron_ingot]).
crafting(diamond_kirk, [stick, diamond]).
crafting(gold_kirk, [stick, gold_ingot]).
crafting(netherite_kirk, [diamond_kirk, netherite_ingot]).
crafting(iron_ingot, [iron_ore]).
crafting(gold_ingot, [gold_ore]).
crafting(netherite_ingot, [netherite_scrap, gold_ingot]).
crafting(netherite_scrap, [ancient_blocks]).
crafting(wood_boards, [wood]).
crafting(stick, [wood_boards]).

% -----------------------------
% Сложное рекурсивное правило для проверки, что можно скрафтить, имея текущие ресурсы.
% -----------------------------

% Правило для определения, что можно скрафтить, имея текущие ресурсы.
can_craft(Tool, Resources) :- 
    crafting(Tool, Items), 
    can_craft_items(Items, Resources).

% Правило для проверки, что все элементы списка могут быть скрафчены
can_craft_items([], _).
can_craft_items([Item|Rest], Resources) :- 
    member(Item, Resources), 
    can_craft_items(Rest, Resources).
can_craft_items([Item|Rest], Resources) :- 
    can_craft(Item, Resources), 
    can_craft_items(Rest, Resources).

% -----------------------------
% Правило для проверки, что можно добыть block, имея tool
% -----------------------------
can_mine(Block, Tool) :- 
    block(Block), 
    tool(Tool), 
    damage(Tool, Damage), 
    mining(Block, MinDamage), 
    Damage >= MinDamage.

% -----------------------------
% Правило подсчёта времени, необходимых для добычи block'а (если блок нельзя добыть - false)
% -----------------------------
time_to_mine(Block, Tool, Hits) :- 
    block(Block), 
    tool(Tool), 
    can_mine(Block, Tool), 
    endurance(Block, Endurance), 
    mining(Block, MinDamage),
    Hits is Endurance / MinDamage.

% -----------------------------
% Правило для поиска всех block'ов, которые можно добыть с tool'ом
% -----------------------------

blocks_mineable_with(Tool, Blocks) :-
    findall(Block, can_mine(Block, Tool), Blocks).

% -----------------------------
% Правило для поиска всех tool'ов, которыми можно добыть block
% -----------------------------
tools_to_mine(Block, Tools) :-
    findall(Tool, can_mine(Block, Tool), Tools).
