from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.utils import is_request_type
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from ask_sdk_core.skill_builder import SkillBuilder

class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        print("Inside LaunchRequestHandler")
        speech_text = "<speak>Here is an example of a speechcon. <say-as interpret-as='interjection'>abracadabra!</say-as>.</speak>"
        
        return (
            handler_input.response_builder
                .speak(speech_text)
                .ask(speak_output)
                .response
        )

sb = SkillBuilder()
sb.add_request_handler(LaunchRequestHandler())
lambda_handler = sb.lambda_handler()