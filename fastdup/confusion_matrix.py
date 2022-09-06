import warnings
import numpy as np

"""
The below code is taken from scikit-learn, using NEW BSD license.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

def column_or_1d(y, *, warn=False):
    """Ravel column or 1d numpy array, else raises an error.
    Parameters
    ----------
    y : array-like
       Input data.
    warn : bool, default=False
       To control display of warnings.
    Returns
    -------
    y : ndarray
       Output data.
    Raises
    ------
    ValueError
        If `y` is not a 1D array or a 2D array with a single row or column.
    """
    y = np.asarray(y)
    shape = np.shape(y)
    if len(shape) == 1:
        return np.ravel(y)
    if len(shape) == 2 and shape[1] == 1:
        return np.ravel(y)

    raise ValueError(
        "y should be a 1d array, got an array of shape {} instead.".format(shape)
    )


def unique_labels(truth, pred):
    total = truth.copy()
    total.extend(pred)
    return np.array(sorted(np.unique(total)))





def multilabel_confusion_matrix(
        y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False
):
    """Compute a confusion matrix for each class or sample.
    .. versionadded:: 0.21
    Compute class-wise (default) or sample-wise (samplewise=True) multilabel
    confusion matrix to evaluate the accuracy of a classification, and output
    confusion matrices for each class or sample.
    In multilabel confusion matrix :math:`MCM`, the count of true negatives
    is :math:`MCM_{:,0,0}`, false negatives is :math:`MCM_{:,1,0}`,
    true positives is :math:`MCM_{:,1,1}` and false positives is
    :math:`MCM_{:,0,1}`.
    Multiclass data will be treated as if binarized under a one-vs-rest
    transformation. Returned confusion matrices will be in the order of
    sorted unique labels in the union of (y_true, y_pred).
    Read more in the :ref:`User Guide <multilabel_confusion_matrix>`.
    Parameters
    ----------
    y_true : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Ground truth (correct) target values.
    y_pred : {array-like, sparse matrix} of shape (n_samples, n_outputs) or \
            (n_samples,)
        Estimated targets as returned by a classifier.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    labels : array-like of shape (n_classes,), default=None
        A list of classes or column indices to select some (or to force
        inclusion of classes absent from the data).
    samplewise : bool, default=False
        In the multilabel case, this calculates a confusion matrix per sample.
    Returns
    -------
    multi_confusion : ndarray of shape (n_outputs, 2, 2)
        A 2x2 confusion matrix corresponding to each output in the input.
        When calculating class-wise multi_confusion (default), then
        n_outputs = n_labels; when calculating sample-wise multi_confusion
        (samplewise=True), n_outputs = n_samples. If ``labels`` is defined,
        the results will be returned in the order specified in ``labels``,
        otherwise the results will be returned in sorted order by default.
    See Also
    --------
    confusion_matrix : Compute confusion matrix to evaluate the accuracy of a
        classifier.
    Notes
    -----
    The `multilabel_confusion_matrix` calculates class-wise or sample-wise
    multilabel confusion matrices, and in multiclass tasks, labels are
    binarized under a one-vs-rest way; while
    :func:`~sklearn.metrics.confusion_matrix` calculates one confusion matrix
    for confusion between every two classes.
    Examples
    --------
    Multilabel-indicator case:
    >>> import numpy as np
    >>> from sklearn.metrics import multilabel_confusion_matrix
    >>> y_true = np.array([[1, 0, 1],
    ...                    [0, 1, 0]])
    >>> y_pred = np.array([[1, 0, 0],
    ...                    [0, 1, 1]])
    >>> multilabel_confusion_matrix(y_true, y_pred)
    array([[[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[1, 0],
            [0, 1]],
    <BLANKLINE>
           [[0, 1],
            [1, 0]]])
    Multiclass case:
    >>> y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    >>> y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    >>> multilabel_confusion_matrix(y_true, y_pred,
    ...                             labels=["ant", "bird", "cat"])
    array([[[3, 1],
            [0, 2]],
    <BLANKLINE>
           [[5, 0],
            [1, 0]],
    <BLANKLINE>
           [[2, 1],
            [1, 2]]])
    """
    #y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    assert len(y_true) == len(y_pred)
    if len(labels) == 2:
        y_type = 'binary'
    else:
        y_type = 'multiclass'

    present_labels = unique_labels(y_true, y_pred)
    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack(
            [labels, np.setdiff1d(present_labels, labels, assume_unique=True)]
        )



    label_dict = {label: i for i, label in enumerate(labels)}
    #print('label_dict', label_dict)
    y_true = [label_dict[label] for label in y_true]
    y_pred = [label_dict[label] for label in y_pred]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # labels are now from 0 to len(labels) - 1 -> use bincount
    tp = np.array(y_true) == np.array(y_pred)
    #print('tp',tp)
    tp_bins = y_true[tp]
    #print(tp_bins)
    #print(y_true)
    #print(y_pred)

    tp_bins_weights = None

    if len(tp_bins):
        tp_sum = np.bincount(
            tp_bins, weights=tp_bins_weights, minlength=len(labels)
        )
    else:
        # Pathological case
        true_sum = pred_sum = tp_sum = np.zeros(len(labels))
    if len(y_pred):
        pred_sum = np.bincount(y_pred, weights=sample_weight, minlength=len(labels))
    if len(y_true):
        true_sum = np.bincount(y_true, weights=sample_weight, minlength=len(labels))

    # Retain only selected labels
    indices = np.searchsorted(present_labels, labels[:n_labels])
    tp_sum = tp_sum[indices]
    true_sum = true_sum[indices]
    pred_sum = pred_sum[indices]

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum

    if sample_weight is not None and samplewise:
        sample_weight = np.array(sample_weight)
        tp = np.array(tp)
        fp = np.array(fp)
        fn = np.array(fn)
        tn = sample_weight * y_true.shape[1] - tp - fp - fn
    elif sample_weight is not None:
        tn = sum(sample_weight) - tp - fp - fn
    elif samplewise:
        tn = y_true.shape[1] - tp - fp - fn
    else:
        tn = y_true.shape[0] - tp - fp - fn

    return np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)

def _check_set_wise_labels(y_true, y_pred, average, labels, pos_label):
    """Validation associated with set-wise metrics.
    Returns identified labels.
    """
    average_options = (None, "micro", "macro", "weighted", "samples")
    if average not in average_options and average != "binary":
        raise ValueError("average has to be one of " + str(average_options))

    if len(labels) == 2:
        y_type = 'binary'
    else:
        y_type = 'multiclass'

    #y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    # Convert to Python primitive type to avoid NumPy type / Python str
    # comparison. See https://github.com/numpy/numpy/issues/6784
    present_labels = unique_labels(y_true, y_pred).tolist()
    if average == "binary":
        if y_type == "binary":
            if pos_label not in present_labels:
                if len(present_labels) >= 2:
                    raise ValueError(
                        f"pos_label={pos_label} is not a valid label. It "
                        f"should be one of {present_labels}"
                    )
            labels = [pos_label]
        else:
            average_options = list(average_options)
            if y_type == "multiclass":
                average_options.remove("samples")
            raise ValueError(
                "Target is %s but average='binary'. Please "
                "choose another average setting, one of %r." % (y_type, average_options)
            )
    elif pos_label not in (None, 1):
        pass
        # warnings.warn(
        #     "Note that pos_label (set to %r) is ignored when "
        #     "average != 'binary' (got %r). You may use "
        #     "labels=[pos_label] to specify a single positive class."
        #     % (pos_label, average),
        #     UserWarning,
        #     )
    return labels



def _prf_divide(
        numerator, denominator, metric, modifier, average, warn_for, zero_division="warn"
):
    """Performs division and handles divide-by-zero.
    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn" or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and "f-score" in warn_for:
        msg_start = "{0} and F-score are".format(metric.title())
    elif metric in warn_for:
        msg_start = "{0} is".format(metric.title())
    elif "f-score" in warn_for:
        msg_start = "F-score is"
    else:
        return result

    #_warn_prf(average, modifier, msg_start, len(result))

    return result



def precision_recall_fscore_support(
        y_true,
        y_pred,
        *,
        beta=1.0,
        labels=None,
        pos_label=1,
        average=None,
        warn_for=("precision", "recall", "f-score"),
        sample_weight=None,
        zero_division="warn",
):
    """Compute precision, recall, F-measure and support for each class.
    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label a negative sample as
    positive.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The F-beta score can be interpreted as a weighted harmonic mean of
    the precision and recall, where an F-beta score reaches its best
    value at 1 and worst score at 0.
    The F-beta score weights recall more than precision by a factor of
    ``beta``. ``beta == 1.0`` means recall and precision are equally important.
    The support is the number of occurrences of each class in ``y_true``.
    If ``pos_label is None`` and in binary classification, this function
    returns the average precision, recall and F-measure if ``average``
    is one of ``'micro'``, ``'macro'``, ``'weighted'`` or ``'samples'``.
    Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    beta : float, default=1.0
        The strength of recall versus precision in the F-score.
    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    pos_label : str or int, default=1
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.
    average : {'binary', 'micro', 'macro', 'samples', 'weighted'}, \
            default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division:
           - recall: when there are no positive labels
           - precision: when there are no positive predictions
           - f-score: both
        If set to "warn", this acts as 0, but warnings are also raised.
    Returns
    -------
    precision : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Precision score.
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        Recall score.
    fbeta_score : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
        F-beta score.
    support : None (if average is not None) or array of int, shape =\
        [n_unique_labels]
        The number of occurrences of each label in ``y_true``.
    Notes
    -----
    When ``true positive + false positive == 0``, precision is undefined.
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, as will f-score,
    and ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    References
    ----------
    .. [1] `Wikipedia entry for the Precision and recall
           <https://en.wikipedia.org/wiki/Precision_and_recall>`_.
    .. [2] `Wikipedia entry for the F1-score
           <https://en.wikipedia.org/wiki/F1_score>`_.
    .. [3] `Discriminative Methods for Multi-labeled Classification Advances
           in Knowledge Discovery and Data Mining (2004), pp. 22-30 by Shantanu
           Godbole, Sunita Sarawagi
           <http://www.godbole.net/shantanu/pubs/multilabelsvm-pakdd04.pdf>`_.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import precision_recall_fscore_support
    >>> y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
    >>> y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
    >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
    (0.22..., 0.33..., 0.26..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
    (0.33..., 0.33..., 0.33..., None)
    >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
    (0.22..., 0.33..., 0.26..., None)
    It is possible to compute per-label precisions, recalls, F1-scores and
    supports instead of averaging:
    >>> precision_recall_fscore_support(y_true, y_pred, average=None,
    ... labels=['pig', 'dog', 'cat'])
    (array([0.        , 0.        , 0.66...]),
     array([0., 0., 1.]), array([0. , 0. , 0.8]),
     array([2, 2, 2]))
    """
    #_check_zero_division(zero_division)
    #if beta < 0:
    #    raise ValueError("beta should be >=0 in the F-beta score")
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)

    # Calculate tp_sum, pred_sum, true_sum ###
    samplewise = average == "samples"
    MCM = multilabel_confusion_matrix(
        y_true,
        y_pred,
        sample_weight=sample_weight,
        labels=labels,
        samplewise=samplewise,
    )
    tp_sum = MCM[:, 1, 1]
    pred_sum = tp_sum + MCM[:, 0, 1]
    true_sum = tp_sum + MCM[:, 1, 0]

    if average == "micro":
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta**2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        tp_sum, pred_sum, "precision", "predicted", average, warn_for, zero_division
    )
    recall = _prf_divide(
        tp_sum, true_sum, "recall", "true", average, warn_for, zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    # if zero_division == "warn" and ("f-score",) == warn_for:
    #     if (pred_sum[true_sum == 0] == 0).any():
    #         _warn_prf(average, "true nor predicted", "F-score is", len(true_sum))

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.0] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == "weighted":
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = np.float64(1.0)
            if zero_division in ["warn", 0]:
                zero_division_value = np.float64(0.0)
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            if pred_sum.sum() == 0:
                return (
                    zero_division_value,
                    zero_division_value,
                    zero_division_value,
                    None,
                )
            else:
                return (np.float64(0.0), zero_division_value, np.float64(0.0), None)

    elif average == "samples":
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != "binary" or len(precision) == 1
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = None  # return no support

    return precision, recall, f_score, true_sum


def classification_report(
        y_true,
        y_pred,
        *,
        labels=None,
        target_names=None,
        sample_weight=None,
        digits=2,
        output_dict=False,
        zero_division="warn",
):
    """Build a text report showing the main classification metrics.
    Read more in the :ref:`User Guide <classification_report>`.
    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array-like of shape (n_labels,), default=None
        Optional list of label indices to include in the report.
    target_names : list of str of shape (n_labels,), default=None
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    digits : int, default=2
        Number of digits for formatting output floating point values.
        When ``output_dict`` is ``True``, this will be ignored and the
        returned values will not be rounded.
    output_dict : bool, default=False
        If True, return output as dict.
        .. versionadded:: 0.20
    zero_division : "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division. If set to
        "warn", this acts as 0, but warnings are also raised.
    Returns
    -------
    report : str or dict
        Text summary of the precision, recall, F1 score for each class.
        Dictionary returned if output_dict is True. Dictionary has the
        following structure::
            {'label 1': {'precision':0.5,
                         'recall':1.0,
                         'f1-score':0.67,
                         'support':1},
             'label 2': { ... },
              ...
            }
        The reported averages include macro average (averaging the unweighted
        mean per label), weighted average (averaging the support-weighted mean
        per label), and sample average (only for multilabel classification).
        Micro average (averaging the total true positives, false negatives and
        false positives) is only shown for multi-label or multi-class
        with a subset of classes, because it corresponds to accuracy
        otherwise and would be the same for all metrics.
        See also :func:`precision_recall_fscore_support` for more details
        on averages.
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    See Also
    --------
    precision_recall_fscore_support: Compute precision, recall, F-measure and
        support for each class.
    confusion_matrix: Compute confusion matrix to evaluate the accuracy of a
        classification.
    multilabel_confusion_matrix: Compute a confusion matrix for each class or sample.
    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                  precision    recall  f1-score   support
    <BLANKLINE>
         class 0       0.50      1.00      0.67         1
         class 1       0.00      0.00      0.00         1
         class 2       1.00      0.67      0.80         3
    <BLANKLINE>
        accuracy                           0.60         5
       macro avg       0.50      0.56      0.49         5
    weighted avg       0.70      0.60      0.61         5
    <BLANKLINE>
    >>> y_pred = [1, 1, 0]
    >>> y_true = [1, 1, 1]
    >>> print(classification_report(y_true, y_pred, labels=[1, 2, 3]))
                  precision    recall  f1-score   support
    <BLANKLINE>
               1       1.00      0.67      0.80         3
               2       0.00      0.00      0.00         0
               3       0.00      0.00      0.00         0
    <BLANKLINE>
       micro avg       1.00      0.67      0.80         3
       macro avg       0.33      0.22      0.27         3
    weighted avg       1.00      0.67      0.80         3
    <BLANKLINE>
    """

    #y_type, y_true, y_pred = _check_targets(y_true, y_pred)


    if labels is None:
        labels = unique_labels(y_true, y_pred)
        labels_given = False
    else:
        labels = np.asarray(labels)
        labels_given = True


    if len(labels) == 2:
        y_type = 'binary'
    else:
        y_type = 'multiclass'

    # labelled micro average
    micro_is_accuracy = (y_type == "multiclass" or y_type == "binary") and (
            not labels_given or (set(labels) == set(unique_labels(y_true, y_pred)))
    )

    if target_names is not None and len(labels) != len(target_names):
        if labels_given:
            warnings.warn(
                "labels size, {0}, does not match size of target_names, {1}".format(
                    len(labels), len(target_names)
                )
            )
        else:
            raise ValueError(
                "Number of classes, {0}, does not match size of "
                "target_names, {1}. Try specifying the labels "
                "parameter".format(len(labels), len(target_names))
            )
    if target_names is None:
        target_names = ["%s" % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]
    # compute per-class results without averaging
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        sample_weight=sample_weight,
        zero_division=zero_division,
    )
    rows = zip(target_names, p, r, f1, s)

    if y_type.startswith("multilabel"):
        average_options = ("micro", "macro", "weighted", "samples")
    else:
        average_options = ("micro", "macro", "weighted")

    if output_dict:
        report_dict = {label[0]: label[1:] for label in rows}
        for label, scores in report_dict.items():
            report_dict[label] = dict(zip(headers, [i.item() for i in scores]))
    else:
        longest_last_line_heading = "weighted avg"
        name_width = max(len(cn) for cn in target_names)
        width = max(name_width, len(longest_last_line_heading), digits)
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"
        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"
        for row in rows:
            report += row_fmt.format(*row, width=width, digits=digits)
        report += "\n"

    # compute all applicable averages
    for average in average_options:
        if average.startswith("micro") and micro_is_accuracy:
            line_heading = "accuracy"
        else:
            line_heading = average + " avg"

        # compute averages with specified averaging method
        avg_p, avg_r, avg_f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=average,
            sample_weight=sample_weight,
            zero_division=zero_division,
        )
        avg = [avg_p, avg_r, avg_f1, np.sum(s)]

        if output_dict:
            report_dict[line_heading] = dict(zip(headers, [i.item() for i in avg]))
        else:
            if line_heading == "accuracy":
                row_fmt_accuracy = (
                        "{:>{width}s} "
                        + " {:>9.{digits}}" * 2
                        + " {:>9.{digits}f}"
                        + " {:>9}\n"
                )
                report += row_fmt_accuracy.format(
                    line_heading, "", "", *avg[2:], width=width, digits=digits
                )
            else:
                report += row_fmt.format(line_heading, *avg, width=width, digits=digits)

    if output_dict:
        if "accuracy" in report_dict.keys():
            report_dict["accuracy"] = report_dict["accuracy"]["precision"]
        return report_dict
    else:
        return report
