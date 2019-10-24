from typing import Dict, Optional

from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util

@Model.register("category_crisis_lossweight_classifier")
class CategoryCrisisClassifierWithLossWeight(Model):
    """
    This ``Model`` performs crisis-related tweet classification for it's emergency categories.  We assume we're given a
    raw tweet's text, and we predict its output labels (categories).

    The basic model structure: we'll embed the text and encode it with
    a Seq2VecEncoder, and then pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    text_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the text to a vector.
    classifier_feedforward : ``FeedForward``
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 text_encoder: Seq2VecEncoder,
                 threshold: float,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:

        super(CategoryCrisisClassifierWithLossWeight, self).__init__(vocab, regularizer)
        self.threshold = threshold
        self.text_field_embedder = text_field_embedder
        self.title_encoder = text_encoder
        self.classifier_feedforward = classifier_feedforward

        if text_field_embedder.get_output_dim() != text_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            text_encoder.get_input_dim()))

        # loss weights correspond to the order of information types as follows
        # {0: 'Sentiment', 1: 'Hashtags', 2: 'News', 3: 'Irrelevant', 4: 'MultimediaShare', 5: 'ThirdPartyObservation',
        #  6: 'FirstPartyObservation', 7: 'Factoid', 8: 'Discussion', 9: 'OriginalEvent', 10: 'Location', 11: 'Advice',
        #  12: 'ContextualInformation', 13: 'Weather', 14: 'EmergingThreats', 15: 'ServiceAvailable', 16: 'Donations',
        #  17: 'Official', 18: 'NewSubEvent', 19: 'InformationWanted', 20: 'SearchAndRescue', 21: 'MovePeople',
        #  22: 'CleanUp', 23: 'Volunteer', 24: 'GoodsServices'}

        pos_weights=torch.tensor([ 1.0000,  1.0869,  1.2438,  1.5013,  1.5451,  1.9002,  1.9796,  2.4450,
         3.3531,  3.6609,  4.7413,  4.8119,  5.1208,  5.5894,  7.0552,  8.4015,
        11.2066, 11.6695, 13.9506, 14.0638, 17.4586, 20.5034, 24.5000, 28.9487,
        32.3684], dtype=torch.float32)

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weights)
        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.LongTensor],
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:

        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self.text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        encoded_text = self.title_encoder(embedded_text, text_mask)

        logits = self.classifier_feedforward(encoded_text)
        output_dict = {'logits': logits}

        if labels is not None:
            output_dict["loss"] = self.loss(logits, labels.float())
        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities
        predictions = class_probabilities.cpu().data.numpy()
        return_lables = []
        for probs in predictions:
            labels = []
            for i, v in enumerate(probs):
                if float(v) > self.threshold:
                    labels.append(self.vocab.get_token_from_index(i, 'labels'))
            return_lables.append(labels)
        output_dict['labels'] = return_lables
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy(TODO:customize for MLC)": 0.2}

    # def _mapping_function(f_value,f_lambda=0.48):
    #     return (1/(1-f_lambda))*(1/(1+math.exp(-f_value+1))-f_lambda)
    #
    # def _mapWeights(input_weights):
    #     out_weights=[(1/_mapping_function(1))*_mapping_function(e) for e in input_weights.numpy()]
    #     return torch.tensor(out_weights)
