import kivy
from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


class CameraExample(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Create camera object
        self.cameraObject = Camera(play=False)
        self.cameraObject.play = True
        self.cameraObject.resolution = (300, 300)

        # Button that captures pictures
        self.cameraClick = Button(text="Take Photo")
        self.cameraClick.size_hint(.5, .2)
        self.cameraClick.pos_hint = {'x': .25, 'y':.75}

        # Bind the button's on_press to onCameraClick
        self.cameraClick.bind(on_press=self.onCameraClick)

        # Append camera and button to the layout
        layout.add_widget(self.cameraObject)
        layout.add_widget(self.cameraClick)

        # return the root widget
        return layout

    # Take the current frame of the video as the photograph
    def onCameraClick(self, *args):
        self.cameraObject.export_to_png('/kivyexamples/selfie.png')


if __name__ == '__main__':
    CameraExample().run()
