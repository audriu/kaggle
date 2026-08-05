import os
import time
from collections import defaultdict

from cg.api import (
    AreaType,
    Card,
    CardType,
    EnergyType,
    Observation,
    OptionType,
    Pokemon,
    SelectContext,
    all_card_data,
    search_begin,
    search_end,
    search_release,
    search_step,
    to_observation_class,
)

"""
Mega Lucario ex Deck — enhanced
Official TPC rule-based sample + damage fidelity + look-ahead search.
"""

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
BASIC_POKEMON_IDS = [
    c.cardId for c in all_card if c.basic and c.cardType == CardType.POKEMON
]

Makuhita = 673
Hariyama = 674
Lunatone = 675
Solrock = 676
Riolu = 677
Mega_Lucario_ex = 678
Dusk_Ball = 1102
Switch = 1123
Premium_Power_Pro = 1141
Fighting_Gong = 1142
Poke_Pad = 1152
Hero_Cape = 1159
Boss_Orders = 1182
Carmine = 1192
Lillie_Determination = 1227
Gravity_Mountain = 1252
Basic_Fighting_Energy = 6

SEARCH_NODE_BUDGET = 72
SEARCH_TIME_BUDGET_S = 0.40


class AttackPlan:
    attacker = -1
    target = -1
    attack_index = -1
    remain_hp = -1
    energy = False


plan = AttackPlan()
pre_turn = 0
ability_used = False
deck_remaining = defaultdict(int)
serial_seen = set()


def reset_deck_track():
    global deck_remaining, serial_seen
    deck_remaining = defaultdict(int)
    for cid in my_deck:
        deck_remaining[cid] += 1
    serial_seen = set()


def consume_card(card):
    if card is None:
        return
    if card.serial not in serial_seen:
        serial_seen.add(card.serial)
        if deck_remaining[card.id] > 0:
            deck_remaining[card.id] -= 1
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            consume_card(c)
        for c in card.tools:
            consume_card(c)
        for c in card.preEvolution:
            consume_card(c)


def sync_deck_track(obs, my_index):
    reset_deck_track()
    st = obs.current
    me = st.players[my_index]
    for zone in (me.hand, me.discard, me.bench, me.active):
        for c in zone:
            consume_card(c)
    for c in st.stadium:
        consume_card(c)
    if st.looking:
        for c in st.looking:
            consume_card(c)
    if obs.select and obs.select.effect:
        consume_card(obs.select.effect)
    if obs.select and obs.select.deck:
        for c in obs.select.deck:
            consume_card(c)


def get_card(obs: Observation, area: AreaType, index: int, player_index: int):
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


def prize_count(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    for card in pokemon.energyCards:
        if card.id == 12:
            count -= 1
    for card in pokemon.tools:
        if card.id == 1172 and "Lillie" in data.name:
            count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon) -> int:
    data = card_table[pokemon.id]
    score = prize_count(pokemon) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    if pokemon.id in (144, 322, 323, 337):
        score -= 200
    if pokemon.id == 112 and len(pokemon.energies) >= 1:
        score += 300
    score += pokemon.hp
    return score


def evaluate_state(obs, my_index):
    st = obs.current
    if st is None:
        return 0.0
    if st.result >= 0:
        if st.result == my_index:
            return 1_000_000.0
        if st.result == 2:
            return 0.0
        return -1_000_000.0
    me = st.players[my_index]
    op = st.players[1 - my_index]
    score = (6 - len(me.prize)) * 5000.0 - (6 - len(op.prize)) * 5000.0
    for p in list(me.active) + list(me.bench):
        if p is None:
            continue
        score += 200 + len(p.energies) * 80 + p.hp * 0.4
        if p.id == Mega_Lucario_ex:
            score += 350
        elif p.id in (Solrock, Lunatone, Hariyama):
            score += 100
    for p in list(op.active) + list(op.bench):
        if p is None:
            continue
        score -= 120 + len(p.energies) * 40 + prize_count(p) * 70
    if me.hand:
        score += len(me.hand) * 12
    return score


def fill_hidden(obs):
    st = obs.current
    me = st.yourIndex
    opp = 1 - me
    your_deck = []
    for cid, n in deck_remaining.items():
        your_deck.extend([cid] * max(0, n))
    dc = st.players[me].deckCount
    if len(your_deck) < dc:
        your_deck.extend([Basic_Fighting_Energy] * (dc - len(your_deck)))
    your_deck = your_deck[:dc]
    your_prize = [Basic_Fighting_Energy] * len(st.players[me].prize)
    basic = BASIC_POKEMON_IDS[0] if BASIC_POKEMON_IDS else 22
    odc = st.players[opp].deckCount
    opponent_deck = ([basic] + [Basic_Fighting_Energy] * max(0, odc - 1)) if odc else []
    opponent_prize = [Basic_Fighting_Energy] * len(st.players[opp].prize)
    opponent_hand = [
        basic if i == 0 else Basic_Fighting_Energy
        for i in range(st.players[opp].handCount)
    ]
    opponent_active = []
    if st.players[opp].active and st.players[opp].active[0] is None:
        opponent_active = [basic]
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def search_refine(obs, scores, heuristic_pick):
    if (
        obs.search_begin_input is None
        or obs.select is None
        or obs.select.context != SelectContext.MAIN
        or obs.select.maxCount != 1
        or len(obs.select.option) > 18
        or obs.current is None
        or obs.current.turn < 2
    ):
        return heuristic_pick

    my_index = obs.current.yourIndex
    start = time.perf_counter()
    nodes = 0
    try:
        root = search_begin(obs, *fill_hidden(obs))
    except Exception:
        try:
            search_end()
        except Exception:
            pass
        return heuristic_pick

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidates = []
    for i in ranked:
        if len(candidates) >= 7:
            break
        if scores[i] < -1 and (not heuristic_pick or i != heuristic_pick[0]):
            continue
        candidates.append(i)
    if heuristic_pick and heuristic_pick[0] not in candidates:
        candidates.insert(0, heuristic_pick[0])

    best_act = heuristic_pick
    best_val = -1e18
    heur_val = None

    try:
        for idx in candidates:
            if nodes >= SEARCH_NODE_BUDGET or (time.perf_counter() - start) > SEARCH_TIME_BUDGET_S:
                break
            try:
                node = search_step(root.searchId, [idx])
                nodes += 1
            except Exception:
                continue
            val = evaluate_state(node.observation, my_index)
            cur = node
            depth = 0
            while (
                depth < 4
                and cur.observation.select is not None
                and cur.observation.current
                and cur.observation.current.result < 0
                and nodes < SEARCH_NODE_BUDGET
                and (time.perf_counter() - start) < SEARCH_TIME_BUDGET_S
            ):
                sel = cur.observation.select
                move = None
                for j, o in enumerate(sel.option):
                    if o.type == OptionType.ATTACK:
                        move = [j]
                        break
                if move is None:
                    # Prefer non-END MAIN actions with positive heuristic bias via type
                    for j, o in enumerate(sel.option):
                        if o.type in (
                            OptionType.ABILITY,
                            OptionType.EVOLVE,
                            OptionType.ATTACH,
                            OptionType.PLAY,
                        ):
                            move = [j]
                            break
                if move is None:
                    k = max(sel.minCount, min(1, sel.maxCount))
                    move = list(range(k)) if k else []
                try:
                    nxt = search_step(cur.searchId, move)
                    nodes += 1
                    search_release(cur.searchId)
                    cur = nxt
                    val = 0.65 * val + 0.35 * evaluate_state(cur.observation, my_index)
                    depth += 1
                    if move and sel.context == SelectContext.MAIN:
                        if sel.option[move[0]].type in (OptionType.ATTACK, OptionType.END):
                            val = evaluate_state(cur.observation, my_index)
                            break
                except Exception:
                    break
            val += scores[idx] * 0.025
            if heuristic_pick and idx == heuristic_pick[0]:
                heur_val = val
            if val > best_val:
                best_val = val
                best_act = [idx]
            try:
                search_release(cur.searchId)
            except Exception:
                pass
    finally:
        try:
            search_end()
        except Exception:
            pass

    if heur_val is not None and best_act != heuristic_pick:
        if best_val >= heur_val + 350:
            return best_act
        return heuristic_pick
    return best_act


def agent(obs_dict: dict) -> list[int]:
    global plan, pre_turn, ability_used

    select_raw = obs_dict.get("select")
    if select_raw is None:
        reset_deck_track()
        pre_turn = 0
        return my_deck

    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        options = select_raw.get("option") or []
        max_count = int(select_raw.get("maxCount", 1))
        min_count = int(select_raw.get("minCount", 1))
        k = min(max(min_count, min(max_count, len(options))), len(options))
        return list(range(k))

    if obs.select is None:
        reset_deck_track()
        pre_turn = 0
        return my_deck

    if obs.current is None:
        k = max(obs.select.minCount, min(obs.select.maxCount, len(obs.select.option)))
        return list(range(k))

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    my_prize = len(my_state.prize)

    sync_deck_track(obs, my_index)

    if pre_turn != state.turn:
        pre_turn = state.turn
        plan = AttackPlan()
        ability_used = False

    field_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    discard_counts = defaultdict(int)

    attacker1 = False
    attacker2 = False
    for card in my_state.active + my_state.bench:
        if card is None:
            continue
        field_counts[card.id] += 1
        if card.id in (Makuhita, Hariyama) and len(card.energies) >= 3:
            attacker2 = True
        elif card.id in (Riolu, Mega_Lucario_ex) and len(card.energies) >= 2:
            attacker1 = True

    for card in my_state.hand:
        hand_counts[card.id] += 1
    for card in my_state.discard:
        discard_counts[card.id] += 1

    stadium_id = state.stadium[0].id if state.stadium else 0

    can_attack = False
    if context == SelectContext.MAIN:
        can_switch = False
        can_op_switch = False
        can_use_mega_brave = False
        for o in select.option:
            if o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Switch:
                    can_switch = True
                elif card.id == Boss_Orders:
                    can_op_switch = True
            elif o.type == OptionType.EVOLVE:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                if card.id == Hariyama:
                    can_op_switch = True
            elif o.type == OptionType.RETREAT:
                can_switch = True
            elif o.type == OptionType.ATTACK:
                can_attack = True
                if o.attackId == 983:
                    can_use_mega_brave = True

        my_cards = [my_state.active[0]]
        for pokemon in my_state.bench:
            my_cards.append(pokemon)
        op_cards = [op_state.active[0]]
        for pokemon in op_state.bench:
            op_cards.append(pokemon)

        ppp_avail = hand_counts[Premium_Power_Pro] > 0

        if state.turn >= 2:
            best_score = -1
            for i, my_pokemon in enumerate(my_cards):
                if my_pokemon is None:
                    continue
                if i != 0 and not can_switch:
                    break
                for a in range(2):
                    energy_required = 0
                    base_damage = 0
                    base_score = 0
                    ignore_wr = False
                    if my_pokemon.id == Mega_Lucario_ex:
                        if a == 0:
                            energy_required = 1
                            base_damage = 130
                            base_score += 60 * min(3, discard_counts[Basic_Fighting_Energy])
                        else:
                            energy_required = 2
                            base_damage = 270
                        if my_prize == 2 or my_prize == 3:
                            base_score -= 500
                    elif a == 1:
                        break
                    elif my_pokemon.id == Hariyama:
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Makuhita:
                        for o in select.option:
                            if o.type == OptionType.EVOLVE:
                                index = o.inPlayIndex
                                if o.inPlayArea == AreaType.BENCH:
                                    index += 1
                                if index == i:
                                    break
                        else:
                            break
                        base_score -= 100
                        energy_required = 3
                        base_damage = 210
                    elif my_pokemon.id == Solrock:
                        if field_counts[Lunatone] >= 1:
                            energy_required = 1
                            base_damage = 70
                            ignore_wr = True

                    if base_damage <= 0:
                        continue

                    more_energy = False
                    energy_count = len(my_pokemon.energies)
                    if a == 1 and i == 0 and energy_count >= 2 and not can_use_mega_brave:
                        break
                    if energy_count < energy_required:
                        if hand_counts[Basic_Fighting_Energy] >= 1 and not state.energyAttached:
                            energy_count += 1
                            if energy_count < energy_required:
                                continue
                            more_energy = True
                        else:
                            continue

                    for j, op_pokemon in enumerate(op_cards):
                        if op_pokemon is None:
                            continue
                        if j != 0 and not can_op_switch:
                            break

                        # Damage with optional PPP; Gravity Mountain vs Stage 2
                        def apply_dmg(base, use_ppp):
                            dmg = base + (30 if use_ppp else 0)
                            hp = op_pokemon.hp
                            data = card_table[op_pokemon.id]
                            if stadium_id == Gravity_Mountain and data.stage2:
                                hp = max(0, hp - 30)
                            if not ignore_wr:
                                if data.weakness == EnergyType.FIGHTING:
                                    dmg *= 2
                                elif data.resistance == EnergyType.FIGHTING:
                                    dmg -= 30
                            return max(0, dmg), hp

                        dmg0, hp = apply_dmg(base_damage, False)
                        dmg1, _ = apply_dmg(base_damage, True)
                        damage = dmg0
                        bonus = 0
                        if hp > dmg0 and ppp_avail and hp <= dmg1:
                            damage = dmg1
                            bonus = 90
                        prize = 0
                        score = pokemon_score(op_pokemon)
                        if hp <= damage:
                            prize = prize_count(op_pokemon)
                        else:
                            score *= damage / max(1, hp)
                        score += base_score + bonus

                        if len(op_state.prize) <= prize:
                            score = 50000

                        if i == 0:
                            score += 220
                        if j == 0:
                            score += 300
                        score += energy_count
                        if best_score < score:
                            best_score = score
                            plan.attacker = i
                            plan.target = j
                            plan.attack_index = a
                            plan.remain_hp = hp - dmg0  # without PPP (official semantics)
                            plan.energy = more_energy

    def energy_score(pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        score = 8000
        if active:
            score += 10
        if pokemon.id in (Makuhita, Hariyama):
            if pokemon.id == Hariyama:
                score += 1
            if energy_count < 3:
                score += 100
            if attacker2:
                score -= 50
        elif pokemon.id == Lunatone:
            score -= 100
        elif pokemon.id == Solrock:
            if energy_count < 1:
                score += 20
            else:
                score -= 100
        elif pokemon.id in (Riolu, Mega_Lucario_ex):
            if pokemon.id == Mega_Lucario_ex:
                score += 1
            if energy_count < 2:
                score += 100
            if attacker1:
                score -= 50
        return score

    scores = []
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card is not None:
                energy_count = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                if context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                    if o.playerIndex == my_index:
                        score += energy_count * 2
                        if o.index == plan.attacker - 1:
                            score += 100
                        if card.id == Mega_Lucario_ex:
                            score += 8 if my_prize in (2, 3) else 20
                        elif card.id == Hariyama and energy_count >= 2:
                            score += 15
                        elif card.id == Makuhita and energy_count >= 2:
                            score += 10
                        elif card.id == Solrock:
                            score += 5
                        elif card.id == Riolu:
                            score += 4
                    else:
                        if o.index == plan.target - 1:
                            score += 100
                elif context == SelectContext.SETUP_ACTIVE_POKEMON:
                    if card.id == Solrock:
                        score = 2 if state.firstPlayer == my_index else 4
                    elif card.id == Riolu:
                        score = 3
                    elif card.id == Makuhita:
                        score = 1
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if card.id == Riolu:
                        score = 30 if field_counts[Riolu] + field_counts[Mega_Lucario_ex] < 2 else -5
                    elif card.id == Lunatone:
                        score = 28 if field_counts[Lunatone] < 1 else -20
                    elif card.id == Solrock:
                        score = 25 if field_counts[Solrock] < 1 else -20
                    elif card.id == Makuhita:
                        score = 12
                    else:
                        score = 1
                elif context == SelectContext.TO_HAND:
                    score = 200 - hand_counts[card.id] * 100
                    if card.id == Makuhita:
                        score += -10 if field_counts[card.id] >= 1 else 10
                    elif card.id == Hariyama:
                        score += 20 if field_counts[Makuhita] >= 1 else -20
                    elif card.id == Lunatone:
                        score += -250 if field_counts[card.id] >= 1 else 60
                    elif card.id == Solrock:
                        score += -250 if field_counts[card.id] >= 1 else 50
                    elif card.id == Riolu:
                        n = field_counts[card.id] + field_counts[Mega_Lucario_ex]
                        if n >= 2:
                            score -= 150
                        elif n >= 1:
                            score -= 3
                        else:
                            score += 40
                    elif card.id == Mega_Lucario_ex:
                        score += 40 if field_counts[Riolu] >= 1 else -15
                    elif card.id == Basic_Fighting_Energy:
                        score += 30 if (not ability_used or not state.energyAttached) else -1
                elif context == SelectContext.ATTACH_FROM:
                    score = energy_score(card, o.area == AreaType.ACTIVE)
                elif context == SelectContext.DISCARD:
                    data = card_table[card.id]
                    if data.cardType == CardType.BASIC_ENERGY:
                        score = 40
                    elif card.id in (Carmine, Lillie_Determination):
                        score = 15
                    else:
                        score = -5
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[card.id]
            if data.cardType == CardType.POKEMON:
                score = 20000
                if card.id in (Lunatone, Solrock) and field_counts[card.id] >= 1:
                    score = -1
                elif card.id == Riolu and field_counts[card.id] + field_counts[Mega_Lucario_ex] >= 2:
                    score = -1
            else:
                score = 10000
                if card.id == Switch:
                    score = 6000 if plan.attacker > 0 else -1
                elif card.id == Premium_Power_Pro:
                    if state.supporterPlayed and plan.remain_hp <= 0:
                        score = -1
                    elif not can_attack:
                        if (
                            not state.supporterPlayed
                            and hand_counts[Carmine] > 0
                            and hand_counts[Lillie_Determination] == 0
                        ):
                            score = 3050
                        else:
                            score = -1
                    else:
                        score = 5000
                elif card.id == Boss_Orders:
                    score = 3200 if plan.target >= 1 else -1
                elif card.id == Carmine:
                    score = 3000
                elif card.id == Lillie_Determination:
                    score = 3100
                elif card.id == Gravity_Mountain:
                    op_has_s2 = any(
                        p is not None and card_table[p.id].stage2
                        for p in (op_state.active + op_state.bench)
                    )
                    if stadium_id == Gravity_Mountain:
                        score = -1
                    elif op_has_s2:
                        score = 4200
                    elif stadium_id == 0:
                        score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if card.id == Hero_Cape:
                score = 7000
                if pokemon.id == Riolu:
                    score += 100
                elif pokemon.id == Mega_Lucario_ex:
                    score += 200
            else:
                score = energy_score(pokemon, o.inPlayArea == AreaType.ACTIVE)
                if o.inPlayArea == AreaType.ACTIVE:
                    if plan.attacker == 0 and plan.energy:
                        score += 200
                else:
                    if plan.attacker == 1 + o.inPlayIndex and plan.energy:
                        score += 200
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = 9000 + len(pokemon.energies)
            if pokemon.id == Makuhita and plan.target == 0:
                score = -1
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == 1267:
                score = 1
            else:
                score = 30000
        elif o.type == OptionType.RETREAT:
            score = 2000 if plan.attacker >= 1 else -1
        elif o.type == OptionType.ATTACK:
            score = 1000
            if plan.attack_index == 1:
                if o.attackId == 983:
                    score += 100
            else:
                if o.attackId != 983:
                    score += 100

        scores.append(score)

    # Prefer skipping negative optional multi-picks (bench setup)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    output = []
    for i, idx in enumerate(ranked):
        if i >= select.maxCount:
            break
        if scores[idx] >= 0 or i < select.minCount:
            output.append(idx)
        elif context not in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH):
            output.append(idx)
    if len(output) < select.minCount:
        for idx in ranked:
            if idx not in output:
                output.append(idx)
            if len(output) >= select.minCount:
                break
    output = output[: select.maxCount]

    if context == SelectContext.MAIN and output:
        o = select.option[output[0]]
        if o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if card.id == Lunatone:
                ability_used = True
        output = search_refine(obs, scores, output)

    return output
