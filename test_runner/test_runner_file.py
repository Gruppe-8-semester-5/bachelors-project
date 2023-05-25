# Either set dict, or everything else.
from algorithms import gradient_descent_template
from models.utility import accuracy, make_mini_batch_gradient


class Runner:
    def __init__(
        self,
        w0=None,
        alg=None,
        model=None,
        epsilon=None,
        max_iter=None,
        GD_params=None,
        batch=None,
        dic=None,
    ) -> None:
        if dic is None and w0 is None:
            raise Exception("You must give some arguments...")
        if dic is None:
            self.dict = {
                "w0": w0,
                "alg": alg,
                "model": model,
                "epsilon": epsilon,
                "max_iter": max_iter,
                "GD_params": GD_params,
                batch: batch,
            }
        else:
            self.dict = dic
            if "test_set" not in dic:
                self.dict["test_set"] = None
            if "batch" not in dic:
                self.dict["batch"] = batch
        self.res = None

    def run(self):
        if self.res is None:
            res = {}
            for x in unpack_generator(self.dict):
                res[str(x)] = (x, actual_run(x))
            self.res = res

    def get_result(self, *args, **kwargs):
        return [x for _, x in self.get_res_and_description(*args, **kwargs)]

    def get_res_and_description(
        self,
        dic=None,
        alg=None,
        model=None,
        epsilon=None,
        max_iter=None,
        w0=None,
        GD_params=None,
        batch=None,
    ):
        if not self.res:
            self.run()
        res = []
        if dic is None:
            # Copy
            dic = self.dict | {}
        if alg is not None:
            dic["alg"] = alg
        if epsilon is not None:
            dic["epsilon"] = epsilon
        if max_iter is not None:
            dic["max_iter"] = max_iter
        if model is not None:
            dic["model"] = model
        if w0 is not None:
            dic["w0"] = w0
        if GD_params is not None:
            dic["GD_params"] = GD_params
        if batch is not None:
            dic["batch"] = batch
        for x in unpack_generator(dic):
            # print(self.res)
            # print(str(x))
            res.append(self.res[str(x)])
        return res


def is_simple(val):
    return not (isinstance(val, list) | isinstance(val, dict))


def unpack_generator(dic, acc={}) -> dict:
    if not dic:
        yield acc
    else:
        name = next(iter(sorted(dic.keys())))
        val = dic[name]
        del dic[name]
        if is_simple(val):
            cur_acc = acc | {name: val}
            yield from unpack_generator(dic, cur_acc)
        elif isinstance(val, list):
            for i in val:
                cur_acc = acc | {name: i}
                gen = unpack_generator(dic, cur_acc)
                yield from gen
        elif isinstance(val, dict):
            gen = unpack_generator(val, {})
            for i in gen:
                cur_acc = acc | {name: i}
                yield from unpack_generator(dic, cur_acc)
        dic[name] = val


def actual_run(dic):
    algo = dic["alg"](**dic["GD_params"])
    (X, y) = dic["data_set"]
    model = dic["model"]
    if dic["test_set"] is not None:
        (X_test, y_test) = dic["test_set"]
        acc = lambda w: accuracy(y_test, make_predictions(w, X_test, model.predict))
    else:
        acc = lambda w: accuracy(y, make_predictions(w, X, model.predict))

    batch = dic["batch"]
    gradient = model.gradient
    complete_gradient = None
    if batch is not None:
        grad = make_mini_batch_gradient(X, y, batch, gradient)
        complete_gradient = lambda w: gradient(X, y, w)
    else:
        grad = lambda w: gradient(X, y, w)
    gradient_and_loss = lambda w: model.gradient_and_loss(X, y, w)


    auto_stop = True
    if "auto_stop" in dic:
        auto_stop = dic["auto_stop"]

    if "serialize" in dic:
        serialize = dic["serialize"]
    else:
        serialize = True

    return gradient_descent_template.find_minima(
        start_weights=dic["w0"],
        algorithm=algo,
        derivation=grad,
        gradient_and_loss=gradient_and_loss,
        epsilon=dic["epsilon"],
        max_iter=dic["max_iter"],
        accuracy=acc,
        auto_stop=auto_stop,
        complete_derivation=complete_gradient,
        serialize=serialize,
    )


def make_predictions(weights, data, predictor):
    # predictions = []
    return predictor(weights, data)
    # return predictions
