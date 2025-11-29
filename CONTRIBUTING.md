# Contributing to the Wine Classification Project
This outlines how to propose a change to the Wine Classification Project.

## Fixing typos
Currently, only the project team is allowed direct editing access. The documentation is reviewed frequently and errors are updated when encountered. For any errors that need immediate attention, please notify the team of the issue.

## Pull request process
Before you make a substantial pull request, you should always file an issue and make sure someone from the team agrees that it's a problem. If you've found a bug, create an associated issue.

We recommend that you create a Git branch for each pull request (PR). Please see the documentation GitHub has provided for instruction on [how to create a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).

## Developer notes
For developer contributions, these are the requirements and instructions.

### Developer dependencies
* `conda` (version 25.11.0 or higher)
* `conda-lock` (version 3.0.4 or higher)

### Adding additional dependencies
There is an automated workflow that will run when new dependencies are added.
1. Create a new branch with a name relevant to these dependency updates. And ensure you are on that branch.
2. Add all additional dependencies to the `environment.yml` file.
3. Run the following command `conda-lock -k explicit --file environment.yml -p linux-64`. This will update the `conda-linux-64.lock` file.
4. To ensure the Docker image will build and runs, re-build the image based on these changes locally.
5. If everything builds correctly, push the changes to GitHub.
6. The push will trigger a workflow to re-build the Docker image, publish it to Docker Hub, and update the `docker-compose.yml` file with the new image tag. Check this is successful.
7. Create a Pull Request to merge the changes made on this branch to that of the main branch.

## Code of Conduct
Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Attribution
These contributing guidelines were adapted from the [dplyr contributing guidelines](https://github.com/tidyverse/dplyr/blob/main/.github/CONTRIBUTING.md).
