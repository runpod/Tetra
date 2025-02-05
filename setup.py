# setup.py
import setuptools

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()

setuptools.setup(
    name="tetra",                  # Replace with your own package name
    version="0.0.0b1",                         # Package version
    author="Marut Pandya",
    author_email="pandyamarut@gmail.com",
    description="Execute functions remotely",
    long_description=long_description,       # From README.md
    # long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/your-repo",  # Project URL
    packages=setuptools.find_packages(),     # Automatically find sub-packages
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
