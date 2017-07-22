Installing Orb
##############

Unpacking
---------

Once the package is downloaded you can uncompress it wherever you want
but you want to make it accessible from python. To do it you have
plenty of choices but the more commons depends on whether you are an
administrator and you want to make it accessible by any user or you
are a user and you want to install it for you alone.

Make ORB accessible to all users
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are an administator (and you have administration rights), just
create the file /usr/local/lib/python2.7/dist-packages/orb.pth and add
the path to the extracted module

  /path/to/orbs/module

you can also untar the whole module in
/usr/local/lib/python2.7/dist-packages/.


.. note:: It is recommended to create a symbolic link to use the
     scripts from any location. e.g.::

       sudo ln -s /link/to/orbs /usr/bin/orbs
       sudo ln -s /link/to/orbs-optcreator /usr/bin/orbs-optcreator

Make ORB accessible to you alone
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this case just create the file ~/.local/lib/python2.7/dist-packages/orb.pth and add the path to the extracted module

  /path/to/orbs/module

you can also untar the whole module in
~/.local/lib/python2.7/dist-packages/.

.. note:: It is recommended to create a symbolic link to use the
     scripts from any location. e.g.::

       sudo ln -s /link/to/orbs ~/.local/bin/orbs
       sudo ln -s /link/to/orbs-optcreator ~/.local/bin/orbs-optcreator
