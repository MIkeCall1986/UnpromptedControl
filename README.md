# UnpromptedControl

**By sponsoring me, you're not just supporting my work - you're helping to create a more collaborative, innovative open source community 💖 [sponsor](https://github.com/sponsors/vijishmadhavan?o=sd&sc=t).**

[Get more updates on Twitter](https://twitter.com/Vijish68859437)

ControlNet is a highly regarded tool for guiding StableDiffusion models, and it has been widely acknowledged for its effectiveness. In this repository, A simple hack that allows for the restoration or removal of objects without requiring user prompts. By leveraging this approach, the workflow can be significantly streamlined, leading to enhanced process efficiency.

## No-prompt

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/vijishmadhavan/UnpromptedControl/blob/master/UnpromptedControl.ipynb)

![restore Result](examples/eg2gif.gif)
![restore Result](examples/objgif.gif)
## Image Restoration 

In this image restoration is accomplished using the controlnet-canny and stable-diffusion-2-inpainting techniques, with only "" blank input prompts. Additionally, for automatic scratch segmentation, the FT_Epoch_latest.pt model is being used. However, if the segmentation output is not satisfactory, it is possible to manually sketch and refine the mask to achieve better results. As ControlNet model is trained on pairs of images, one of which has missing parts, and it learns to predict the missing parts based on the content of the complete image.

![restore Result](examples/eg1.jpg)

![restore Result](examples/eg2.jpg)

## Object Removal

Automatically removing objects from images is a challenging task that requires a combination of computer vision and deep learning techniques. This code leverages the power of OpenCV inpainting, deep learning-based image restoration, and blending techniques to achieve this task automatically, without the need for user prompts. The ControlNetModel and StableDiffusionInpaintPipeline models play a crucial role in guiding the inpainting process and restoring the image to a more natural-looking state. Overall, this code provides an efficient and effective way to remove unwanted objects from images and produce natural-looking results that are consistent with the surrounding image content. 

**"Surely, it has its limitations and might fail with certain images, especially those of faces, and may require some back and forth. To obtain good results, we need to mask not only the object but also its shadow."**


![restore Result](examples/obj2.jpg)
![restore Result](examples/obj1.jpg)

## Limitation

- Limited Generalization: The algorithm currently has limitations when it comes to processing images of people's faces and bodies. It may not work as expected for these types of images, and additional work is needed to improve its performance in these areas.

- When it comes to removing an object from an image, it's important to consider the surrounding environment and any elements that may be affected by the removal process. In some cases, removing an object may require the removal of a large area surrounding the object, including its shadows.

- To obtain good results, we need to mask not only the object but also its shadow.

## Acknowledgements

https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life (Segmentation)

https://huggingface.co/thibaud/controlnet-sd21

https://github.com/lllyasviel/ControlNet

18.02.26
Ось результати аналізу та стратегія трансформації для проекту **UnpromptedControl**, підготовлені у форматі для копіювання в Notion.

---

# 📑 Звіт AI-консультанта: Проект "UnpromptedControl"

**UnpromptedControl** — це спеціалізований інструмент на базі ControlNet, призначений для автоматичного видалення об'єктів та реставрації зображень без необхідності введення текстових підказок (prompts).

## 🧬 Частина 1: "ДНК" Проекту

Логіку коду проекту можна розбити на такі **атомарні функції**:

*   **Автоматична сегментація (`scratch_detection.py`):** Використання моделі `FT_Epoch_latest.pt` для виявлення дефектів (подряпин) на зображенні.
*   **Інпеїнтинг на базі ControlNet (`ControlNetModel`):** Основна функція, що використовує `controlnet-canny` та `stable-diffusion-2-inpainting` для заповнення відсутніх частин зображення.
*   **Управління конвеєром (`StableDiffusionInpaintPipeline`):** Оркестрація процесу відновлення контенту на основі навчених пар зображень.
*   **Гібридна обробка (OpenCV + Deep Learning):** Поєднання традиційних методів інпеїнтингу OpenCV з глибинним навчанням та техніками змішування (blending) для досягнення природного вигляду.
*   **Інтерфейс виконання (`demo.py` / `UnpromptedControl.ipynb`):** Скрипти для запуску процесу обробки, генерації масок та отримання фінального результату.

### 💎 Головна технічна цінність
Головна цінність проекту полягає в **усуненні потреби у текстових промптах**. Це значно спрощує робочий процес (workflow), оскільки система сама прогнозує відсутні частини на основі контексту зображення, що робить процес редагування ефективнішим та доступнішим для автоматизації.

---

## 🚀 Частина 2: "Трансформація" (Інтеграція з Gemini LLM)

Додавання мультимодальної моделі як **Gemini** (через **GitHub Models**) перетворює проект із вузькоспеціалізованого скрипта на інтелектуальну систему редагування.

### Як зміниться функціонал?
1.  **Семантичне виявлення об'єктів:** Gemini зможе ідентифікувати об'єкти за описом (наприклад, "видали сміття на фоні"), автоматично створюючи точні маски без ручного малювання.
2.  **Вирішення проблеми тіней:** Джерела вказують на необхідність маскування тіней для кращого результату. Gemini може автоматично визначати межі об'єкта разом із його тінню, що зараз є технічним обмеженням.
3.  **Контекстне відновлення:** LLM допоможе системі краще "розуміти", що саме має бути на місці видаленого об'єкта (наприклад, продовження цегляної стіни або текстура трави), надаючи точніші параметри для ControlNet.

### Сценарій сервісу "Auto-Restore Web" (UnpromptedControl + Gemini + ID_{$})

Створення сервісу "Ремонт фото в один клік" на вашому сайті:
1.  **Завантаження:** Користувач завантажує старе або пошкоджене фото на сайт.
2.  **Аналіз (Gemini):** Через API Gemini аналізує зображення, виявляє подряпини або небажані об'єкти та генерує координати для маскування.
3.  **Оркестрація (ID_{$}):** Ваш базовий Python-скрипт `ID_{$}` отримує дані від Gemini, формує маску та запускає ядро `UnpromptedControl` для обробки.
4.  **Фіналізація:** Скрипт `ID_{$}` застосовує фінальне змішування кольорів та повертає очищене зображення користувачу.
5.  **Деплой:** Використовуючи **GitHub Spark**, ви створюєте та розгортаєте цей інтелектуальний додаток з мінімальними зусиллями.

---

## 📋 План дій для Notion
| Крок | Дія | Результат |
| :--- | :--- | :--- |
| **1** | Встановлення залежностей та моделей ControlNet | Працездатне ядро обробки |
| **2** | Підключення Gemini через **GitHub Models** | "Розумна" сегментація об'єктів |
| **3** | Створення Python-містка (`ID_{$}`) | Автоматизація між сайтом та AI-моделлю |
| **4** | Розгортання інтерфейсу через **GitHub Spark** | Готовий сервіс на вашому сайті |

---

### 💡 Резюме

**Суть:** **Автоматичне видалення об'єктів без текстових промптів**.

**AI-Роль:** **Створення інтелектуальних застосунків через Spark**.
