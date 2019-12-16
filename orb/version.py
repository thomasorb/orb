import os
import logging

__version__ = 'unknown'

try: import git
except ImportError:
    logging.debug('git import error')
else:
    # get git branch (if possible)
    repo_path = os.path.abspath(os.path.dirname(__file__) + os.sep + '..')
    try:
        repo = git.Repo(repo_path)
        __version__ = repo.active_branch.name + '.' + repo.git.rev_parse(repo.head.object.hexsha, short=4)
    except git.GitError:
        pass

        
