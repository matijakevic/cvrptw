use rand::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::hash::Hash;
use std::io::prelude::*;
use std::io::BufReader;
use std::iter;
use std::time::Instant;

#[derive(Debug, PartialEq, Hash, Eq)]
struct Node {
    id: u64,
    x: i64,
    y: i64,
    demand: u64,
    ready_time: u64,
    due_time: u64,
    service_duration: u64,
}

#[derive(Debug)]
struct Instance {
    num_vehicles: u64,
    max_capacity: u64,
    start: Node,
    nodes: Vec<Node>,
}
fn distance(n1: &Node, n2: &Node) -> f64 {
    (((n1.x - n2.x).pow(2) + (n1.y - n2.y).pow(2)) as f64).sqrt()
}

fn load_instance(path: &str) -> Instance {
    let f = File::open(path).expect("Error while reading the instance file");
    let reader = BufReader::new(f);

    let mut lines = reader.lines().skip(2);
    let line = lines.next().unwrap().unwrap();
    let mut data = line
        .split_whitespace()
        .map(str::trim)
        .map(str::parse::<u64>)
        .map(Result::unwrap);

    let mut nodes = lines.skip(4).map(Result::unwrap).map(|line| {
        let mut data = line.split_whitespace().map(str::trim);

        Node {
            id: data.next().unwrap().parse::<u64>().unwrap(),
            x: data.next().unwrap().parse::<i64>().unwrap(),
            y: data.next().unwrap().parse::<i64>().unwrap(),
            demand: data.next().unwrap().parse::<u64>().unwrap(),
            ready_time: data.next().unwrap().parse::<u64>().unwrap(),
            due_time: data.next().unwrap().parse::<u64>().unwrap(),
            service_duration: data.next().unwrap().parse::<u64>().unwrap(),
        }
    });

    Instance {
        num_vehicles: data.next().unwrap(),
        max_capacity: data.next().unwrap(),
        start: nodes.next().unwrap(),
        nodes: nodes.collect(),
    }
}

fn calculate_route_penalty(instance: &Instance, route: &[&Node]) -> f64 {
    if route.is_empty() {
        return 0.0;
    }

    let mut penalty = 0.0;

    let weight = route.iter().map(|route| route.demand).sum::<u64>();

    if weight > instance.max_capacity {
        penalty += weight as f64 - instance.max_capacity as f64;
    }

    let mut time = instance.start.ready_time as f64;
    let mut prev_node = &instance.start;

    for node in route.iter().chain(iter::once(&&instance.start)) {
        time += distance(prev_node, node).ceil();

        if time > node.due_time as f64 {
            penalty += time - node.due_time as f64;
        }

        if time < node.ready_time as f64 {
            time = node.ready_time as f64;
        }

        time += node.service_duration as f64;
        prev_node = node;
    }

    return penalty;
}

fn calculate_route_distance(instance: &Instance, route: &[&Node]) -> f64 {
    let mut dist = 0f64;
    let mut prev_node = &instance.start;

    for node in route.iter().chain(iter::once(&&instance.start)) {
        dist += distance(prev_node, node);
        prev_node = node;
    }

    return dist;
}

fn calculate_total_distance(instance: &Instance, solution: &[Vec<&Node>]) -> f64 {
    solution
        .iter()
        .map(|route| calculate_route_distance(instance, route))
        .sum()
}

fn calculate_num_vehicles_required(solution: &[Vec<&Node>]) -> usize {
    solution.iter().filter(|route| route.len() != 0).count()
}

fn calculate_penalty(instance: &Instance, solution: &[Vec<&Node>]) -> f64 {
    let cnt = calculate_num_vehicles_required(solution);

    (if cnt as u64 > instance.num_vehicles {
        instance.num_vehicles as f64 - cnt as f64
    } else {
        0.0
    }) + solution
        .iter()
        .map(|route| calculate_route_penalty(instance, route))
        .sum::<f64>()
}

static mut evals: usize = 0;

fn solution_score(instance: &Instance, solution: &[Vec<&Node>]) -> (f64, f64, f64) {
    let cnt = calculate_num_vehicles_required(solution);

    let dist = calculate_total_distance(instance, solution);

    let penalty = calculate_penalty(instance, solution);

    unsafe {
        evals += 1;
    }

    (penalty, cnt as f64, dist)
}

fn solution_info(instance: &Instance, solution: &[Vec<&Node>]) -> String {
    format!(
        "{:?} {} {} {}",
        solution_score(instance, solution),
        calculate_num_vehicles_required(solution),
        calculate_total_distance(instance, solution),
        calculate_penalty(instance, solution)
    )
}

fn construct_random_solution<'a>(instance: &'a Instance) -> Vec<Vec<&'a Node>> {
    let mut rng = SmallRng::from_entropy();
    let mut nodes: Vec<&Node> = instance.nodes.iter().collect();

    nodes.shuffle(&mut rng);

    let mut routes = Vec::new();

    let mut last_index = 0usize;

    for i in 0..nodes.len() {
        let part = &nodes[last_index..i + 1];

        if part.iter().map(|node| node.demand).sum::<u64>() > instance.max_capacity {
            routes.push(nodes[last_index..i].to_vec());
            last_index = i;
        }
    }

    if last_index < nodes.len() {
        routes.push(nodes[last_index..].to_vec());
    }

    routes
}

fn branch_route<'a>(solution: &'a [Vec<&'a Node>]) -> impl Iterator<Item = Transformation> + 'a {
    (0..solution.len())
        .filter(move |i| solution[*i].len() > 1)
        .flat_map(move |i| (0..solution[i].len()).map(move |j| Transformation::Branch(i, j)))
}

fn inter_route_move<'a>(
    solution: &'a [Vec<&'a Node>],
) -> impl Iterator<Item = Transformation> + 'a {
    (0..solution.len())
        .flat_map(move |i| (0..solution[i].len()).map(move |j| (i, j)))
        .flat_map(move |(i, j)| {
            (0..solution.len())
                .filter(move |k| *k != i)
                .map(move |k| (i, j, k))
        })
        .flat_map(move |(i, j, k)| {
            (0..solution[k].len() + 1).map(move |l| Transformation::InterMove(i, j, k, l))
        })
}

fn intra_route_swap<'a>(
    solution: &'a [Vec<&'a Node>],
) -> impl Iterator<Item = Transformation> + 'a {
    (0..solution.len())
        .flat_map(move |i| (0..solution[i].len()).map(move |j| (i, j)))
        .flat_map(move |(i, j)| {
            (0..solution[i].len())
                .filter(move |k| *k != j)
                .map(move |k| Transformation::IntraSwap(i, j, k))
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Transformation {
    InterMove(usize, usize, usize, usize),
    IntraSwap(usize, usize, usize),
    Branch(usize, usize),
}

fn apply_transformation(transformation: Transformation, solution: &mut Vec<Vec<&Node>>) {
    match transformation {
        Transformation::InterMove(i1, i2, i3, i4) => {
            let node = solution[i1].remove(i2);
            solution[i3].insert(i4, node);
        }
        Transformation::IntraSwap(i1, i2, i3) => {
            solution[i1].swap(i2, i3);
        }
        Transformation::Branch(i1, i2) => {
            let node = solution[i1].remove(i2);
            solution.push(vec![node]);
        }
    }
}

fn undo_transformation(transformation: Transformation, solution: &mut Vec<Vec<&Node>>) {
    match transformation {
        Transformation::InterMove(i1, i2, i3, i4) => {
            let node = solution[i3].remove(i4);
            solution[i1].insert(i2, node);
        }
        Transformation::IntraSwap(i1, i2, i3) => {
            solution[i1].swap(i2, i3);
        }
        Transformation::Branch(i1, i2) => {
            let v = solution.pop().unwrap();
            solution[i1].insert(i2, v[0]);
        }
    }
}

fn search_space<'a>(solution: &'a [Vec<&'a Node>]) -> Vec<Transformation> {
    intra_route_swap(solution)
        .chain(inter_route_move(solution))
        .chain(branch_route(solution))
        .collect()
}

fn get_transform_target<'a>(
    transformation: Transformation,
    solution: &[Vec<&'a Node>],
) -> &'a Node {
    match transformation {
        Transformation::InterMove(i1, i2, i3, i4) => {
            return solution[i1][i2];
        }
        Transformation::IntraSwap(i1, i2, i3) => {
            return solution[i1][i2];
        }
        Transformation::Branch(i1, i2) => {
            return solution[i1][i2];
        }
    }
}

fn tabu_search<'a>(instance: &'a Instance, solution: &[Vec<&'a Node>]) -> Vec<Vec<&'a Node>> {
    let mut current_best = solution.to_vec();
    let mut current_best_score = solution_score(instance, &current_best);
    let mut current = solution.to_vec();

    let mut tabu = HashMap::new();

    for i in 0..1000 {
        let mut best_score = f64::MAX;
        let mut best_transform = None;

        let mut space = search_space(&current);

        space.retain(|transform| !tabu.contains_key(get_transform_target(*transform, &current)));

        let initial_route_scores: Vec<f64> = current
            .iter()
            .map(|route| {
                calculate_route_penalty(instance, route) * calculate_route_distance(instance, route)
            })
            .collect();

        for transform in space {
            apply_transformation(transform, &mut current);

            let score = match transform {
                Transformation::InterMove(i1, i2, i3, i4) => {
                    let s1 = calculate_route_penalty(instance, &current[i1])
                        * calculate_route_distance(instance, &current[i1]);
                    let s2 = calculate_route_penalty(instance, &current[i3])
                        * calculate_route_distance(instance, &current[i3]);
                    s1 - initial_route_scores[i1] + s2 - initial_route_scores[i3]
                }
                Transformation::IntraSwap(i1, i2, i3) => {
                    let s = calculate_route_penalty(instance, &current[i1])
                        * calculate_route_distance(instance, &current[i1]);
                    s - initial_route_scores[i1]
                }
                Transformation::Branch(i1, i2) => {
                    let s1 = calculate_route_penalty(instance, &current[i1])
                        * calculate_route_distance(instance, &current[i1]);
                    let s2 = calculate_route_penalty(instance, current.last().unwrap())
                        * calculate_route_distance(instance, current.last().unwrap());
                    s1 - initial_route_scores[i1] + s2
                }
            };

            if best_transform.is_none() || score < best_score {
                best_score = score;
                best_transform = Some(transform);
            }

            undo_transformation(transform, &mut current);
        }

        if best_transform.is_none() {
            break;
        }

        tabu.insert(
            get_transform_target(best_transform.unwrap(), &current),
            1000,
        );
        apply_transformation(best_transform.unwrap(), &mut current);
        current = current
            .iter()
            .filter(|route| !route.is_empty())
            .cloned()
            .collect();
        let current_score = solution_score(instance, &current);

        // println!(
        //     "{:?} {} {}",
        //     current_score,
        //     solution_info(instance, &current),
        //     tabu.len()
        // );

        if current_score < current_best_score {
            current_best_score = current_score;
            current_best = current.clone();

            if current_score.0 == 0.0 {
                break;
            }
        }

        tabu.retain(|k, v| *v != 0);

        for (k, v) in tabu.iter_mut() {
            *v -= 1;
        }
    }

    current_best
}

fn destroy_random<'a>(instance: &'a Instance, solution: &mut Vec<Vec<&'a Node>>) -> Vec<&'a Node> {
    let mut v: Vec<&'a Node> = Vec::new();
    let mut rng = SmallRng::from_entropy();

    for route in solution.iter() {
        v.extend(route.iter().choose_multiple(&mut rng, 1));
    }

    v.shuffle(&mut rng);
    v.truncate(solution.len());

    for route in solution.iter_mut() {
        route.retain(|node| !v.contains(node));
    }

    solution.retain(|route| !route.is_empty());

    v
}

fn find_flawed_node<'a>(instance: &'a Instance, route: &[&'a Node]) -> Option<usize> {
    let mut time = instance.start.ready_time as f64;
    let mut prev_node = &instance.start;

    for (index, node) in route
        .iter()
        .cloned()
        .chain(iter::once(&instance.start))
        .enumerate()
    {
        time += distance(prev_node, node).ceil();

        if time > node.due_time as f64 && node != &instance.start {
            return Some(index);
        }

        if time < node.ready_time as f64 {
            time = node.ready_time as f64;
        }

        time += node.service_duration as f64;
        prev_node = node;
    }

    return None;
}

fn destroy<'a>(instance: &'a Instance, solution: &mut Vec<Vec<&'a Node>>) -> Vec<&'a Node> {
    let mut v = Vec::new();

    for route in solution.iter() {
        match find_flawed_node(instance, &route) {
            Some(index) => v.push(route[index]),
            None => {}
        }
    }

    for route in solution.iter_mut() {
        route.retain(|node| !v.contains(node));
    }

    solution.retain(|route| !route.is_empty());

    v
}

fn find_insertion_index<'a>(
    instance: &'a Instance,
    route: &mut Vec<&'a Node>,
    node: &'a Node,
) -> (f64, usize) {
    let mut min_index = 0;
    let mut min_score = f64::INFINITY;

    let length = route.len();

    let initial =
        calculate_route_penalty(instance, route) * calculate_route_distance(instance, route);

    for i in 0..length + 1 {
        route.insert(i, node);

        let score = calculate_route_penalty(instance, route)
            * calculate_route_distance(instance, route)
            - initial;

        if score < min_score {
            min_score = score;
            min_index = i;
        }

        route.remove(i);
    }

    (min_score, min_index)
}

fn repair<'a>(instance: &'a Instance, solution: &mut Vec<Vec<&'a Node>>, nodes: &Vec<&'a Node>) {
    for node in nodes {
        let ip = solution
            .iter_mut()
            .enumerate()
            .map(|(i, route)| (i, find_insertion_index(instance, route, node)))
            .min_by(|r1, r2| r1.1 .0.partial_cmp(&r2.1 .0).unwrap())
            .map(|r| (r.0, r.1 .1));
        if let Some(sol) = ip {
            solution[sol.0].insert(sol.1, node);
        }
    }
}

fn vlns<'a>(instance: &'a Instance, solution: &[Vec<&'a Node>]) -> Vec<Vec<&'a Node>> {
    let mut current = solution.to_vec();
    let mut current_score = solution_score(instance, &current);

    let mut repeats = 0;
    let mut k = 0;

    loop {
        let mut best_candidate = current.clone();
        let mut best_score = solution_score(instance, &best_candidate);

        for _ in 0..10 {
            let mut candidate = current.clone();
            let nodes = if k == 0 {
                destroy(instance, &mut candidate)
            } else {
                destroy_random(instance, &mut candidate)
            };

            repair(instance, &mut candidate, &nodes);
            let score = solution_score(instance, &candidate);

            if score < best_score {
                best_score = score;
                best_candidate = candidate;
            }

            if k == 0 {
                break;
            }
        }

        //println!("{:?} {}", best_score, solution_info(instance, &current));

        if best_score < current_score {
            current_score = best_score;
            current = best_candidate;

            repeats = 0;
            k = 0;
        } else {
            k += 1;

            if k == 2 {
                k = 0;
                repeats += 1;
            }

            if repeats == 10 {
                break;
            }
        }
    }

    current
}

fn route_to_string(instance: &Instance, route: &[&Node]) -> String {
    let mut s = Vec::<String>::new();
    let mut time = instance.start.ready_time as f64;
    let mut prev_node = &instance.start;

    s.push(format!("{}({})", prev_node.id, time));

    for node in route.iter().cloned().chain(iter::once(&instance.start)) {
        time += distance(prev_node, node).ceil();

        if time > node.due_time as f64 {
            unreachable!("Route should be valid!");
        }

        if time < node.ready_time as f64 {
            time = node.ready_time as f64;
        }

        s.push(format!("{}({})", node.id, time as u64));

        time += node.service_duration as f64;
        prev_node = node;
    }

    return s.join("->");
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<String>>();
    let instance = load_instance(&args[0]);
    let time_limit = args[2].parse::<u64>().unwrap();

    let timer = Instant::now();

    let mut best = Vec::new();
    let mut best_score = solution_score(&instance, &best);
    let mut i = 1;

    while timer.elapsed().as_secs() < time_limit {
        println!("{} {}", timer.elapsed().as_secs(), time_limit);
        println!("search {}", i);
        let solution = construct_random_solution(&instance);

        println!("vlns start");
        let solution = vlns(&instance, &solution);
        //println!("vlns done");

        println!("tabu start");
        let solution = tabu_search(&instance, &solution);
        //println!("tabu done");

        let score = solution_score(&instance, &solution);
        //println!("{:?}", score);

        if score.0 == 0.0 {
            if score < best_score || best.is_empty() {
                best_score = score;
                best = solution;
            }
        }

        i += 1;
    }

    let total_distance = calculate_total_distance(&instance, &best);

    let solution_file = &args[1];

    let mut f = File::create(solution_file).unwrap();

    writeln!(f, "{}", best.len());
    for (i, route) in best.into_iter().enumerate() {
        writeln!(f, "{}: {}", i + 1, route_to_string(&instance, &route));
    }
    writeln!(f, "{}", total_distance);

    unsafe {
        println!("{}", evals);
    }
}
